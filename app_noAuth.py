# app.py
import requests,random,time
from bs4 import BeautifulSoup  
import pandas as pd
import numpy as np
import plotly.express as px
import re
import streamlit as st

import plotly.io as pio

# Create and register custom theme
pio.templates["company_theme"] = pio.templates["plotly_white"]

pio.templates["company_theme"].layout.update(
    # Global font and title
    font=dict(size=14, family="Segoe UI", color="#333"),
    title=dict(font=dict(size=18, family="Segoe UI", color="#333")),

    # Axis settings (no labels)
    xaxis=dict(
        showgrid=False,
        tickfont=dict(size=14),
        showline=False,
        zeroline=False
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="rgba(200,200,200,0.3)",
        tickfont=dict(size=14),
        showline=False,
        zeroline=False
    ),

    # Background
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",

    # Margins
    margin=dict(t=50, b=40, l=10, r=10),

    # Legend
    showlegend=False,

    # Hover label styling 🟡
    hoverlabel=dict(
        bgcolor="white",          # background color of tooltip
        bordercolor="#ccc",         # border color
        font=dict(size=16, family="Segoe UI", color="#333"),
        align="left",             # text alignment inside hover box
        namelength=-1             # show full trace name if applicable
    )
)

# Apply globally
pio.templates.default = "company_theme"




# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Samsung Phones Explorer",
    page_icon="📱",
    layout="wide"
)


# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(show_spinner=False, ttl=60*30)
def fetch_and_clean(_status) -> pd.DataFrame:
    from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

    URL = 'https://www.flipkart.com/search?q=mobiles&as=on&as-show=on&otracker=AS_Query_HistoryAutoSuggest_1_1_na_na_na&otracker1=AS_Query_HistoryAutoSuggest_1_1_na_na_na&as-pos=1&as-type=HISTORY&suggestionId=mobiles&requestId=be0e7ab4-14c3-46c0-873c-0797cdc4980c&as-backfill=on&p%5B%5D=facets.network_type%255B%255D%3D5G&p%5B%5D=facets.brand%255B%255D%3DSamsung&p%5B%5D=facets.availability%255B%255D%3DExclude%2BOut%2Bof%2BStock&sort=recency_desc'

    def get_base_url(url):
        parsed = urlparse(url)
        params = parse_qsl(parsed.query)
        keep = [(k, v) for k, v in params if k in ['q', 'sort'] or k.startswith('p')]
        clean_query = urlencode(keep, doseq=True)

        return urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            '',
            clean_query + '&page={}',
            ''
        ))

    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        "Mozilla/5.0 (X11; Linux x86_64)",
    ]

   
    soups = []
    base_url = get_base_url(URL)
    session = requests.Session()
    page = 1

    _status.update(label="🔎 Starting Flipkart scraping...", state="running")

    while True:
        full_url = base_url.format(page)
        headers = {"User-Agent": random.choice(USER_AGENTS)}

        _status.update(label=f"📄 Fetching page {page}...", state="running")
        res = session.get(full_url, headers=headers)

        if res.status_code == 429:
            _status.update(
                label=f"⏳ 429 Too Many Requests — retrying page {page} in 10 seconds...",
                state="running"
            )
            time.sleep(10)
            continue
            
        if res.status_code == 500:
            _status.update(
                label=f"⏳ 500 Server Error — retrying page {page} in 10 seconds...",
                state="running"
            )
            time.sleep(10)
            continue


        if res.status_code != 200:
            _status.update(
                label=f"⚠️ Page {page} returned {res.status_code}. Skipping...",
                state="running"
            )
            page += 1
            time.sleep(3)
            continue

        soup = BeautifulSoup(res.text, "lxml")
        main_container = soup.find('div', class_="DOjaWF gdgoEp")

        if main_container:
            soups.append(main_container)
            _status.update(label=f"✅ Page {page} fetched successfully.", state="running")
        else:
            _status.update(label=f"⚠️ Page {page}: main container not found.", state="running")

        next_button = soup.find('a', string='Next')
        if not next_button:
            _status.update(label="🛑 No more pages. Scraping complete.", state="complete")
            break

        page += 1
        time.sleep(random.uniform(2, 4))





    # From each Main container we have to extract multiple phone containers and add it to a list phone_container. This has details of phone in every pages
    phone_containers=[]

    for s in soups:
        page_items=s.find_all("div", class_="tUxRFH")
        phone_containers.extend(page_items)



    # Even refurbised phones are included in our phone_containers. remove such phones
    filtered_phone_containers = []

    for c in phone_containers:
        name_tag = c.find("div", class_="KzDlHZ")

        name = name_tag.text.strip().lower()

        # must start with 'samsung' and NOT contain 'refurbished'
        if name.startswith("samsung") and "refurbished" not in name:
            filtered_phone_containers.append(c)

    # overwrite
    phone_containers = filtered_phone_containers


    ################################################################################################################

    # name and rating doesnt need find_all since 1 name and 1 rating is present per container
    names=[]
    stars=[]
    for c in phone_containers:
        name = c.find("div", class_="KzDlHZ").text.strip()
        names.append(name)

        star_tag = c.find("div", class_="XQDdHH")
        star =float(star_tag.text.strip()) if star_tag else None

        stars.append(star)



    # seperate color and varient name from name
    models = []
    colors = []

    for c in names:
        parts = c.split('(')
        color = parts[1].split(',')[0]

        model = parts[0].split('5G')[0].strip()
        
        colors.append(color)
        models.append(model)


    # Remove samsung galaxy word from phone names

    models = [
        re.sub(r'(?i)samsung\s*(galaxy|glaxy)?', '', m).strip()
        for m in models
    ]

    # capitalize first letter
    models=[i.capitalize() for i in models]

    #To remove inconsistencies like M55 exynos. Cannot Keep just first word because we have variants like S25 Fe also
    models = [i.replace("exynos", "").strip() for i in models]



    series = [i[0] for i in models if i]  # only take first character if not empty


    def get_model_number(name):
        name = name.strip()
        if not name:  # skip empty or None
            return ''
        
        # Case 1: starts with 'Z' → handle specially (keep first two words, remove trailing digits)
        if name.upper().startswith('Z'):
            parts = name.split()
            model = ' '.join(parts[:2])  # e.g. "Z Fold7"
            return re.sub(r'\d+$', '', model).strip().title()
        
        # Case 2: otherwise → first 3 characters (like S24, A55, M14, etc.)
        return name[:3].capitalize()

    generation = [get_model_number(v) for v in models]

    #######################################################################################

    ram_GB, storage_GB, cameras, displays, batteries, processors = [], [], [], [], [], []

    for c in phone_containers:
        spec_tag = c.find_all("li", class_="J+igdf")
        spec = [li.text for li in spec_tag]

        ram = spec[0].split('|')[0].strip()
        ram=int(ram.split()[0])

        storage = spec[0].split('|')[1].strip()
        value=float(storage.split()[0])

        if 'tb' in storage.lower():
            storage_gb = int(value * 1024)
        else:
            storage_gb = int(value)

        display = spec[1]
        camera = spec[2]
        battery = spec[3]
        processor = spec[4] if len(spec_tag) == 6 else None # Phones with 5 specs doesnt have processor listed

        ram_GB.append(ram)
        storage_GB.append(storage_gb)
        displays.append(display)
        cameras.append(camera)
        batteries.append(battery)
        processors.append(processor)


    # replace 'processor' regardless of case 
    processors = [
        re.sub(r'processor', '', i, flags=re.IGNORECASE).strip() if i else None
        for i in processors
    ]

    # Keep processor name only if it exists and contains any keyword
    keywords = ['snapdragon', 'dimensity', 'exynos','gen','elite']

    processors = [
        p if (
            p and any(k in p.lower() for k in keywords)
        ) 
        else None
        for p in processors
    ]


    #Adding Snapdragon in front of items like 7 Gen 1 since it doesn't contain series name
    processors = [
        f"Snapdragon {p}" if p and p[0].isdigit() else p
        for p in processors
    ]

    #if item contains word elite replace it with Snapdragon 8 Elite
    processors = [
        'Snapdragon 8 Elite' if (p and 'elite' in p.lower()) else p
        for p in processors
    ]


    # Extract chipseries from processora
    chip_series=[]

    for i in processors:
        if not i:
            chip_series.append(None)
            continue

        if 'snapdragon' in i.lower() or 'gen' in i.lower():
            chip_series.append('Snapdragon')
        elif 'dimensity' in i.lower():
            chip_series.append('Dimensity')
        elif 'exynos' in i.lower():
            chip_series.append ('Exynos')
        else:
            None


    # extract part of processor name starting from the chip name so  'MediaTek Dimensity 6100+' becomes Dimensity 6100+. This is for standardisation because sometimes 
    # brand name is written(Mediatek) and sometimes not

    chip_num=[]

    for i in processors:
        if not i:
            chip_num.append(None)
            continue
        
        if 'dimensity' in i.lower():
            idx=i.lower().find('dimensity')
            chip_num.append(i[idx:])
        elif 'snapdragon' in i.lower():
            idx=i.lower().find('snapdragon')
            chip_num.append(i[idx:])
        elif 'exynos' in i.lower():
            idx=i.lower().find('exynos')
            chip_num.append(i[idx:])
        else:
            chip_num.append(i)



    # some items doesnt have model name written just 'Exynos Octa Core'. i dont want such items. so i replace all items where series name is not followed by a number with None
    # So i check if first letter of 2nd word is digit or not

    cleaned = []

    for chip in chip_num:
        if not chip:
            cleaned.append(None)
            continue

        words = chip.split()

        # make sure there is at least a second word
        if len(words) < 2:
            cleaned.append(None)
            continue

        # check if first character of 2nd word is a digit
        if words[1][0].isdigit():
            cleaned.append(chip)
        else:
            cleaned.append(None)



    # i want to trim items liek this 'Snapdragon 888 Octa-Core', 'Dimensity 1080, Octa Core', 'Dimensity 700 (MT6833V)', 'Exynos 1330, Octa Core' etc

    # so i create below logic

    # if item in list contains 'gen' like snapdragon 8 gen 3 , then join first 4 words
    # if item in list contains 'elite' like snapdragon 8 elite , then join first 4 words
    # else join first 2 words

    chip_cleaned = []

    for chip in chip_num:
        if not chip:
            chip_cleaned.append(None)
            continue

        words = chip.split()
        chip_lower = chip.lower()

        if 'gen' in chip_lower:
            cleaned = ' '.join(words[:4])
        elif 'elite' in chip_lower:
            cleaned = ' '.join(words[:3])
        else:
            cleaned = ' '.join(words[:2])

        cleaned = cleaned.strip().rstrip(',')
        chip_cleaned.append(cleaned)

    chip_num=chip_cleaned



    # Extract just number
    batteries=[i.split()[0] for i in batteries]
    batteries=[int(i) for i in batteries]


    # seperate inches and display type
    display_size,display_type=[],[]

    for d in displays:
        display_parts=d.split('(')[1].split(')')
        size=display_parts[0]
        type=display_parts[1].strip()

        display_size.append(size)
        display_type.append(type)
        

    display_size=[i.split()[0] for i in display_size]
    display_size=[float(i) for i in display_size]


    frontcam,rearcam=[],[]

    for c in cameras:
        if '|' in c:
            cam_parts=c.split('|')
            part1=cam_parts[0].strip()
            part2=cam_parts[1].replace('Front Camera','').strip()
            frontcam.append(part2)
            rearcam.append(part1)
        else:
            frontcam.append(None)
            rearcam.append(c)

    rearcam_count = [s.count('+') + 1 for s in rearcam]



    ##################################################################################################

    # from each phone container, there is 1 tag for both rating and review
    r_tag = []

    for c in phone_containers:
        tag = c.find('span', class_='Wphh3N')
        if tag:
            r_tag.append(tag.text.strip())
        else:
            r_tag.append(None)


    for c in phone_containers:
        tag=c.find('span',class_='Wphh3N')
        if tag:
            parts=tag.text.split()


    # Extracting both ratings and reviews
    ratings=[]
    reviews=[]
    for c in phone_containers:
        tag=c.find('span',class_='Wphh3N')

        if tag:
            parts=tag.text.split() # ['103', 'Ratings', '&', '9', 'Reviews']
            part1=int(parts[0].replace(',',''))
            part2=int(parts[3].replace(',',''))
        else:
            part1,part2=0,0

        ratings.append(part1)
        reviews.append(part2)


    ##############################################################################

    ### we have to remove both rs symbol and , to convert to number
    prices = []
    for c in phone_containers:
        tag = c.find('div', class_="Nx9bqj _4b5DiR")
        if tag and tag.text:
            try:
                price = int(tag.text.replace(',', '').replace('₹', '').strip())
            except ValueError:
                price = None
        else:
            price = None
        prices.append(price)



    # we have to remove % off part
    discount_percentage=[]

    for c in phone_containers:
        tag=c.find('div',class_='UkUFwK')

        if tag:
            disc=int(tag.text.replace('% off',''))
            discount_percentage.append(disc)
        else:
            discount_percentage.append(None)






    #################################################################
    # CREATE DATAFRAME
    # ###################################################################                                         

    # Create a dictionary of your lists
    data = {
        'model': models, 
        'generation': generation,
        'series': series,
        'color': colors,
        'chipset': chip_num,
        'chip_series':chip_series,
        'ram_gb':ram_GB,
        'storage_gb':storage_GB,
        'display_size_inch': display_size,
        'display_type': display_type,
        'rearcam_count': rearcam_count,
        'battery_mah': batteries,
        'rating': stars,
        'rating_count': ratings,
        'review_count': reviews,
        'price': prices,
        'discount_percentage': discount_percentage
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    # Remove duplicate rows
    df = df.drop_duplicates(keep='first')

    df1 = df[df['chipset'].isnull()]      # rows where chipset is None / NaN
    df2 = df[df['chipset'].notnull()]    # rows where chipset has a value

    # we want to find phones that are present in both df1 and df2. df1 contains phones where chipset is None and d2 contains phone where chipset is present. 
    # This helps to fillup the missing chipset and chip_series columns in df1 by referring df2
    common_variants = set(df1['model']).intersection(df2['model'])

    # Create lookup Series from df2
    chipset_lookup = df2.dropna(subset=['chipset']).drop_duplicates('model').set_index('model')['chipset']
    chip_series_lookup = df2.dropna(subset=['chip_series']).drop_duplicates('model').set_index('model')['chip_series']

    # Fill missing values in df1 using variant lookup
    df1['chipset'] = df1['chipset'].fillna(df1['model'].map(chipset_lookup))
    df1['chip_series'] = df1['chip_series'].fillna(df1['model'].map(chip_series_lookup))


    # Filling None in chipSeries and chipset columns in df using lookup tables we created
    mask = df['chipset'].isna()

    df.loc[mask, 'chipset'] = df.loc[mask, 'model'].map(chipset_lookup)
    df.loc[mask, 'chip_series'] = df.loc[mask, 'model'].map(chip_series_lookup)



    # Using online we lookup to find cpy and fill remaining None values in chipset column
    chipset_map = {
        'A53':  'Exynos 1280',
        'M06':  'Dimensity 6300',
        'M15':  'Exynos 1330',
        'M16':  'Dimensity 6300',
        'M17': 'Exynos 1330',
        'M32':  'Helio G80',
        'M33':  'Exynos 1280',
        'M34':  'Exynos 1280',
        'M35':  'Exynos 1380',
        'M36':  'Exynos 1380',
        'M55':  'Snapdragon 7 Gen 1',
        'M55s': 'Snapdragon 7 Gen 1',
        'M56':  'Snapdragon 7 Gen 1',
    }

    # --- Function to find chipset from model text ---
    def find_chipset(model):
        model_str = str(model).lower()
        for model_code, chip in chipset_map.items():
            if model_code.lower() in model_str:
                return chip
        return np.nan

    # --- Apply only to rows where chipset is missing ---
    mask_missing = df['chipset'].isna() | (df['chipset'] == '') | (df['chipset'].str.lower() == 'none')

    df.loc[mask_missing, 'chipset'] = (
        df.loc[mask_missing, 'model']
        .apply(find_chipset)
    )


    # We fill None values in chip_series column by extracting first word from chipset column
    df.loc[df['chip_series'].isna(), 'chip_series'] = df['chipset'].str.split().str[0]

    # fill blank discount will 0
    df['discount_percentage'].fillna(value=0,inplace=True)

    # Change 'Display' to 'Unknown'
    df['display_type']=df['display_type'].replace('Display','Unknown')


    # Categorise phone into tiers
    tier_map = {
        'S': 'Premium',
        'Z': 'Premium',
        'A': 'Mid-range',
        'M': 'Budget',
        'F': 'Budget'
    }

    # Add the column first
    df['tier'] = df['series'].map(tier_map)

    # Then move it to 4th position (index 3)
    df.insert(3, 'tier', df.pop('tier'))
    # ----------------------------------------------------





    url = "https://www.antutu.com/en/ranking/soc0.htm"
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, headers=headers)

    soup = BeautifulSoup(response.text, "lxml")

    main_container = soup.find('div', class_='m_l fl')
    name_tags = main_container.find_all('div', class_="cpu-name-box")

    chip_names = []
    for i in name_tags:
        name=i.find('span').text.strip()
        chip_names.append(name)

    chip_names = [" ".join(i.split()[1:]) for i in chip_names]

    score_tags=main_container.find_all('li',class_='blast')

    scores=[]

    for i in score_tags:
        score=int(i.text)
        scores.append(score)
    data={'cpu':chip_names,'antutu_score':scores}
    cpu_df=pd.DataFrame(data)

    cpu_df['cpu_score']=cpu_df['antutu_score']*100/cpu_df['antutu_score'].max()

    df = df.merge(cpu_df, left_on='chipset', right_on='cpu', how='left')

    df=df.drop(columns='cpu')


    df = (
    df.sort_values("price")  # sort so lowest price comes first
      .drop_duplicates(
          subset=[col for col in df.columns if col not in ["price", "discount_percentage"]],
          keep="first"
      )
      .reset_index(drop=True)
)
    
    return df


#----------------------------------------

def apply_filters(df: pd.DataFrame,
                  tier_sel, series_sel, generation_sel, model_sel, chipset_sel,
                  ram_sel, storage_sel, price_range):
    q = df.copy()

    if tier_sel:
        q = q[q['tier'] == tier_sel] 
    if series_sel:
        q = q[q['series'] == series_sel] 
    if generation_sel:
        q = q[q['generation'] == generation_sel] 
    if model_sel:
        q = q[q['model'] == model_sel] 
    if chipset_sel:
        q = q[q['chipset'].isin(chipset_sel)] 

    # Multi-select filters
    if ram_sel:
        q = q[q['ram_gb'].isin(ram_sel)] 
    if storage_sel:
        q = q[q['storage_gb'].isin(storage_sel)] 

    # Add a check to prevent errors if filters make the dataframe empty
    if not q.empty:
        q = q[q['price'].between(price_range[0], price_range[1])]

    return q

def kpi_cards(df: pd.DataFrame):
    # Group by model first to avoid duplicates due to color / RAM / storage variants
    model_stats = (
        df.groupby('model', as_index=False)
          .agg({
                'price': 'mean',    # average price per model
                'rating': 'median'  # median rating per model
            })
    )

    # Then compute global KPIs
    unique_models = model_stats['model'].nunique()
    avg_price = model_stats['price'].mean()        # mean of per-model means
    median_rating = model_stats['rating'].median()   # median of per-model medians
    lowest_price = df.groupby('model')['price'].min().min()  # lowest among all variants

    # Display as metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Models (unique)", unique_models)
    with c2:
        st.metric("Avg Price (₹)", f"{avg_price:,.0f}")
    with c3:
        st.metric("Median Rating", f"{median_rating:.1f}")
    with c4:
        st.metric("Lowest Price (₹)", f"{lowest_price:,.0f}" if len(df) else "—")




# -----------------------------
# Sidebar: Data Input + Filters
# -----------------------------
st.title("📱 Samsung 5G Phones Explorer — Web App")


# Initialize persistent variable
if "df_master" not in st.session_state:
    st.session_state.df_master = None

df_loaded = None

# ✅ Show Data Input only if no data loaded yet
if st.session_state.df_master is None:
        # -----------------------------
    # -----------------------------
    # Description below title (2nd page only)
    # -----------------------------
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #eef2ff, #e0e7ff);
        border-left: 6px solid #4f46e5;
        border-radius: 10px;
        padding: 18px 22px;
        margin: 15px 0 25px 0;
        font-size: 16px;
        line-height: 1.6;
        color: #111827;
    ">
    <h4 style="margin-top: 0; color: #1e3a8a;">📘 About this application</h4>
    This app extracts detailed information about <b>Samsung 5G mobile phones</b> from two sources:
    <ul style="margin-top: 8px;">
        <li>📦 <b>Flipkart</b> — product details, prices, and user ratings</li>
        <li>⚙️ <b>Antutu Benchmark</b> — CPU performance scores</li>
    </ul>
    The app automatically cleans and merges this data to help you explore:
    <ul style="margin-top: 8px;">
        <li>💸 Price distribution and user ratings</li>
        <li>📱 Popular models and key specifications</li>
        <li>🔍 Common device features across series</li>
        <li>🏆 Top 2 recommended products from each series</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)


    with st.sidebar:
        # Use a container to group the action
        with st.container(border=True):
            st.markdown("#### 🚀 **Ready to Scrape?**")
            st.markdown("Click below to find all in-stock Samsung 5G phones on Flipkart.")
            
            # --- The button is full-width within the container ---
            run_btn = st.button(
                "🔍 Fetch Live Data", 
                type="primary", 
                use_container_width=True
            )

        if run_btn:
            # Create progress panel in sidebar
            with st.sidebar:
                scraper_status = st.status("Preparing to scrape...", expanded=True)

            try:
                # Pass status object into the scraper
                df_loaded = fetch_and_clean(scraper_status)

                scraper_status.update(
                    label="✅ Scraping completed successfully!",
                    state="complete"
                )

                st.success(f"Loaded **{len(df_loaded):,} rows** successfully!")

            except Exception as e:
                scraper_status.update(
                    label="❌ Scraping failed.",
                    state="error"
                )
                st.error("An unexpected error occurred:")
                st.exception(e)




        # ---- Developer / Debug Mode  ----
        st.markdown("<div style='margin-top: auto;'></div>", unsafe_allow_html=True)
        with st.expander("⚠️ Developer / Debug Mode", expanded=False):
            uploaded_file = st.file_uploader(
                "📂 Upload CSV File",
                type=["csv"],
                label_visibility="collapsed",
                key="debug_file_uploader"  # <-- Unique key 
            )
            if uploaded_file is not None:
                try:
                    df_loaded = pd.read_csv(uploaded_file)
                    st.success(f"✅ Loaded {len(df_loaded):,} rows successfully (debug mode).")
                except Exception as e:
                    st.error(f"❌ Failed to read CSV: {e}")


    # Persist loaded data
    if df_loaded is not None:
        st.session_state.df_master = df_loaded
        st.rerun()  # ✅ refresh UI to hide sidebar

else:
    # ✅ Once data is loaded, show only a minimal sidebar
    with st.sidebar:
        st.success(f"✅ Data Loaded ({len(st.session_state.df_master):,} rows)")
        if st.button("🔄 Reset / Load New Data"):
            st.session_state.df_master = None
            st.rerun()




# ---- Persist data across reruns ----
if "df_master" not in st.session_state:
    st.session_state.df_master = None

if df_loaded is not None:
    st.session_state.df_master = df_loaded

df_master = st.session_state.df_master

# ---- Guard: Stop if no data ----
if df_master is None:
    st.stop()


# -----------------------------
# Build dynamic filter widgets
# -----------------------------
with st.sidebar:
    st.header("📊 Filter Models")

    df_base = df_master.copy()

    # 1️⃣ Tier filter 
    all_tiers = sorted(df_base['tier'].dropna().unique())
    tier_sel = st.selectbox(
        "Tier", 
        all_tiers, 
        index=None
    )

    if tier_sel:
        df_base = df_base[df_base['tier'] == tier_sel] 

    # 2️⃣ Series filter 
    all_series = sorted(df_base['series'].dropna().unique())
    series_sel = st.selectbox(
        "Series", 
        all_series, 
        index=None
    )

    if series_sel:
        df_base = df_base[df_base['series'] == series_sel] 

    # 3️⃣ Generation filter 
    all_generations = sorted(df_base['generation'].dropna().unique())
    generation_sel = st.selectbox(
        "Generation", 
        all_generations, 
        index=None
    )

    if generation_sel:
        df_base = df_base[df_base['generation'] == generation_sel] 

    all_models = sorted(df_base['model'].dropna().unique())
    model_sel = st.selectbox(
        "Model", 
        all_models,
        index=None
    )

    if model_sel:
        df_base = df_base[df_base['model'] == model_sel]

    # 4️⃣ Chipset filter 
    all_chipsets = sorted(df_base['chipset'].dropna().unique())
    chipset_sel = st.multiselect("Chipset", all_chipsets, [])

    if chipset_sel:
        df_base = df_base[df_base['chipset'].isin(chipset_sel)]

    # --- These are still multi-select ---
    ram_sel = st.multiselect("RAM (GB)", sorted(df_base['ram_gb'].dropna().unique()), [])
    storage_sel = st.multiselect("Storage (GB)", sorted(df_base['storage_gb'].dropna().unique()), [])

    # Remaining numeric filters use dynamic min/max
    if not df_base.empty:
        price_min, price_max = int(df_base['price'].min()), int(df_base['price'].max())
        price_range = st.slider("Price (₹)", price_min, price_max, (price_min, price_max), step=500)
    else:
        # Set a default range if df_base is empty
        price_min, price_max = int(df_master['price'].min()), int(df_master['price'].max())
        price_range = st.slider("Price (₹)", price_min, price_max, (price_min, price_max), step=500)
    

# -----------------------------
# Apply filters
# -----------------------------
df = apply_filters(
    df_master,
    tier_sel, series_sel, generation_sel, model_sel, chipset_sel,
    ram_sel, storage_sel, price_range
)

# -----------------------------
# KPI cards + Dataframe
# -----------------------------
st.subheader("📈 Overview")
kpi_cards(df)

# Download
c_dl1, c_dl2 = st.columns([1, 3])
with c_dl1:
    st.download_button(
        "Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="phones_filtered.csv",
        mime="text/csv"
    )

st.dataframe(
    df,
    use_container_width=True,
    hide_index=True
)

st.markdown("""
    <style>
    /* -------- Increase overall tab size -------- */
    button[data-baseweb="tab"] {
        font-size: 20px !important;        /* bigger text + emoji */
        padding: 18px 30px !important;      /* increase tab height and width */
        height: 65px !important;            /* taller tabs */
        border-radius: 10px 10px 0 0 !important;  /* rounded top edges */
    }

    /* -------- Adjust text inside tab -------- */
    button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] p {
        font-size: 20px !important;        /* larger text */
        font-weight: 600 !important;
        margin-bottom: 0 !important;
    }

    /* -------- Active tab styling -------- */
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #f4f4f4 !important;
        border-bottom: 3px solid #4a90e2 !important;
        color: #333 !important;
    }

    /* -------- Hover effect -------- */
    button[data-baseweb="tab"]:hover {
        background-color: #f9f9f9 !important;
        transition: all 0.2s ease-in-out;
        transform: translateY(-2px);
    }
    </style>
""", unsafe_allow_html=True)


# -----------------------------
# Tabs with visuals
# -----------------------------
tabs = st.tabs([
    "📱 Product Lineup",
    "⚙️ Hardware",
    "⭐ Ratings & Popularity",
    "💸 Price Analysis",
    "🤖 Recommendation System"
])

# ==== TAB 1: Models & Variants ====
with tabs[0]:
    st.subheader("""📱Explore how Samsung's Phone Lineup is Structured to Get Insight into Samsung's Product Diversity""")

    # --- Layout: two charts side-by-side ---
    col1, col2, col3 = st.columns([0.75,1,1])

    # =====================
    # Chart 1: Series-level
    # =====================
    with col1:
            # Group and aggregate
        series_data = (
            df.groupby('series')
            .agg({
                'model': lambda x: ', '.join(sorted(x.unique())),  # collect unique variant names
            })
            .reset_index()
        )

        # Add unique count for bar height
        series_data['Number of models'] = series_data['model'].apply(lambda x: len(x.split(', ')))

        # Create interactive bar chart
        series_fig = px.bar(
            series_data,
            x='series',
            y='Number of models',
            title='Number of Models in Each Series',
            hover_data={'model': True}  # show models on hover
        )

        # Clean layout
        series_fig.update_layout(
            xaxis_title=None,
            yaxis_title=None
        )

        st.plotly_chart(series_fig, use_container_width=True)

    # =====================
    # Chart 2: Generation level
    # =====================
    with col3:

        # Aggregate: collect models per generation
        data_models = (
            df.groupby('generation')
            .agg({
                'model': lambda x: ', '.join(sorted(x.unique()))  # collect all unique models
            })
            .reset_index()
        )

        # Add count of unique models
        data_models['Number of models'] = data_models['model'].apply(lambda x: len(x.split(', ')))

        # Filter out models with only one models
        data_models = data_models[data_models['Number of models'] != 1]

        # Sort descending by number of models
        data_models = data_models.sort_values(by='Number of models', ascending=False)

        # Plot
        fig_models = px.bar(
            data_models,
            x='generation',
            y='Number of models',
            title='Generations with Multiple Models',
            hover_data={'model': True},  # show variant list on hover
            category_orders={'generation': data_models['generation'].tolist()}
        )

        # Style customization
        fig_models.update_traces(
            marker_color='#ff7f0e'
        )

        # Layout cleanup
        fig_models.update_layout(
            xaxis_title=None,
            yaxis_title=None,
            yaxis=dict(tickmode='linear', dtick=1)
        )
        st.plotly_chart(fig_models, use_container_width=True)


    with col2:

        # Series-level counts
        series_counts = (
            df.groupby(['tier', 'series'])['model']
            .nunique()
            .reset_index(name='series_count')
        )

        # Tier-level counts
        tier_counts = (
            df.groupby('tier')['model']
            .nunique()
            .reset_index(name='count')
        )

        # Build hover text for each tier
        hover_data = series_counts.groupby('tier').apply(
            lambda x: '<br>'.join([f"{row.series} Series: {row.series_count}" for _, row in x.iterrows()])
        ).reset_index(name='hover_text')

        # Merge with main data
        data = tier_counts.merge(hover_data, on='tier', how='left')

        # Sort by count descending
        data = data.sort_values(by='count', ascending=False)

        # --- Pie Chart ---
        tier_fig = px.pie(
            data,
            names='tier',
            values='count',
            title='Number of Phones by Tier',
        )

        # Custom hover (series breakdown)
        tier_fig.update_traces(
            textfont_color='white',
            hovertemplate='%{label}<br>%{customdata[0]}<extra></extra>',
            customdata=data[['hover_text']],
            textinfo='label+percent',
            pull=[0.05]*len(data)
        )

        st.plotly_chart(tier_fig, use_container_width=True)



# ==== TAB 2: Specifications ====
with tabs[1]:
    st.subheader("⚙️Explore the Distribution of Key Hardware Specifications Across Devices")

    col1, col2,col3 = st.columns([0.5,1,0.5])

    with col1:

        # Prepare data
        chip_data = df.groupby('chip_series')['model'].nunique().sort_values(ascending=False).reset_index()
        chip_data.columns = ['Chip Series', 'Count']

        # Create interactive bar chart
        chip_fig = px.bar(
            chip_data,
            x='Chip Series',
            y='Count',
            title='Number of Models by Chip Series'
        )

        chip_fig.update_traces(
            marker_color='#88e789'
        )

        chip_fig.update_layout(
            xaxis_title=None,
            yaxis_title=None
        )

        st.plotly_chart(chip_fig, use_container_width=True) 



    with col2:
        displaytype_data=df.groupby('display_type')['model'].nunique().sort_values(ascending=False).reset_index()
        displaytype_data.columns=['display_type','count']

        fig_displaytype=px.bar(displaytype_data,
            x='display_type',
            y='count',
            title='Number of Models by Display type')

        fig_displaytype.update_layout(
            xaxis_title=None,
            yaxis_title=None
    )
        st.plotly_chart(fig_displaytype, use_container_width=True)


    with col3:
        cam_data=df.groupby('rearcam_count')['model'].nunique().reset_index()
        cam_data.columns=['rearcam_count','count']

        cam_data['rearcam_count']=cam_data['rearcam_count'].astype(str)

        cam_fig=px.bar(cam_data,
            x='rearcam_count',
            y='count',
            title='Number of Phones by Rear Camera Count')
        
        cam_fig.update_traces(marker_color='#fbe883')

        cam_fig.update_layout(
            xaxis_title=None,
            yaxis_title=None
    )
        st.plotly_chart(cam_fig, use_container_width=True)



    col4,col5,col6=st.columns(3)

    with col4:
        ram_data = df.groupby('ram_gb')['model'].nunique().reset_index()
        ram_data.columns = ['RAM (GB)', 'Count']

        # make x categorical to remove gaps
        ram_data['RAM (GB)'] = ram_data['RAM (GB)'].astype(str)

        ram_fig = px.bar(
            ram_data,
            x='RAM (GB)',
            y='Count',
            title='Number of Models by RAM Capacity'
        )

        ram_fig.update_traces(
            marker_color='#e9d9bd'
        )

        ram_fig.update_layout(
            xaxis_title=None,
            yaxis_title=None
        )
        st.plotly_chart(ram_fig, use_container_width=True)

    with col5:
        storage_data = df.groupby('storage_gb')['model'].nunique().reset_index()
        storage_data.columns = ['storage', 'model count']
        storage_data['storage'] = storage_data['storage'].astype(str)

        fig_storage = px.bar(
            storage_data,
            x='storage',
            y='model count',
            title='Number of Models by Storage Capacity (GB)'
        )
        fig_storage.update_traces(marker_color='#af8fe9')
        fig_storage.update_layout(xaxis_title=None, yaxis_title=None)
        st.plotly_chart(fig_storage, use_container_width=True)

    with col6:

        # Step 1️⃣: Drop duplicate rows (ignore color or other irrelevant differences)
        df_clean = df.drop_duplicates(subset=['model', 'ram_gb', 'storage_gb'])

        # Step 2️⃣: Group by RAM and Storage, count rows (each row = unique model)
        ram_storage_data = (
            df_clean
            .groupby(['ram_gb', 'storage_gb'])
            .size()  # counts how many unique models per RAM+Storage
            .reset_index(name='Model Count')
            .sort_values(by='Model Count', ascending=False)
        )

        # Step 3️⃣: Combine RAM and Storage into one label (e.g., "8+128")
        ram_storage_data['Config'] = ram_storage_data['ram_gb'].astype(str) + '+' + ram_storage_data['storage_gb'].astype(str)

        # Step 4️⃣: Create bar chart
        ram_storage_fig = px.bar(
            ram_storage_data,
            x='Config',
            y='Model Count',
            title='Number of Models by RAM + Storage Combination'
        )

        # Step 5️⃣: Styling
        ram_storage_fig.update_traces(marker_color='#86c5e8')
        ram_storage_fig.update_layout(
            xaxis_title=None,
            yaxis_title=None,
            showlegend=False
        )

        st.plotly_chart(ram_storage_fig, use_container_width=True)




    col7,col8,col9=st.columns(3)

    with col7:
        battery_data = (
    df.groupby('battery_mah')['model']
    .nunique()
    .reset_index(name='count')
    .sort_values('battery_mah', ascending=True)
)
        battery_data['battery_mah'] = battery_data['battery_mah'].astype(str)

        battery_fig = px.bar(battery_data, x='battery_mah', y='count',
                     title='Number of Models by Battery Capacity (mAh)')
        

        battery_fig.update_traces(marker_color='#069594')

        battery_fig.update_layout(
            xaxis_title=None,
            yaxis_title=None
        )

        st.plotly_chart(battery_fig, use_container_width=True)

    with col8:
        display_data = df.groupby('display_size_inch')['model'].nunique().reset_index()
        display_data.columns = ['display_size_inch', 'count']
        display_data['display_size_inch'] = display_data['display_size_inch'].astype(str)

        fig_display = px.bar(
            display_data,
            x='display_size_inch',
            y='count',
            title='Number of Models by Display Size (Inches)'
        )
        fig_display.update_traces(marker_color='#b28989')
        fig_display.update_layout(xaxis_title=None, yaxis_title=None)
        st.plotly_chart(fig_display, use_container_width=True)



# ==== TAB 3: Ratings & Popularity ====
with tabs[2]:
    st.subheader("⭐ Explore User Ratings and Popularity Trends Across Devices")
    col1,col2 =st.columns(2)

    with col1:
        # Compute average rating per phone
        rating_data = (
            df.groupby('model')['rating']
            .median()
            .reset_index(name='median_rating')
        )

        # Round to 1 decimal and group
        rating_data['rating_group'] = rating_data['median_rating'].round(1)

        # Convert to string to prevent gaps
        rating_data['rating_group_str'] = rating_data['rating_group'].astype(str)

        # Build hover text: list variants under each rating group
        hover_data = (
            rating_data.groupby('rating_group_str')['model']
            .apply(lambda x: '<br>'.join(sorted(x)))
            .reset_index(name='hover_text')
        )

        # Count phones per rating group
        rating_counts = (
            rating_data.groupby(['rating_group', 'rating_group_str'])['model']
            .count()
            .reset_index(name='count')
        )

        # Merge count + hover text
        plot_data = rating_counts.merge(hover_data, on='rating_group_str', how='left')

        # Sort by numeric rating descending
        plot_data = plot_data.sort_values(by='rating_group', ascending=True)

        # --- Plot ---
        rating_fig = px.bar(
            plot_data,
            x='rating_group_str',
            y='count',
            title='Number of Phones by Median Rating',
            category_orders={'rating_group_str': plot_data['rating_group_str'].tolist()}  # lock order
        )

        # Custom hover: list variants
        rating_fig.update_traces(
            hovertemplate='%{customdata[0]}<extra></extra>',
            customdata=plot_data[['hover_text']]
        )

        # Layout cleanup
        rating_fig.update_layout(
            xaxis_title=None,
            yaxis_title=None
        )

        st.plotly_chart(rating_fig, use_container_width=True)


        with col2:
            # Step 1: remove duplicate (variant, rating_count) pairs
            unique_pairs = df.drop_duplicates(subset=['model', 'rating_count'])

            rating_cnt_data=unique_pairs.groupby('model')['rating_count'].sum().sort_values(ascending=False).reset_index()

            rating_cnt_data.columns = ['model', 'Total_ratings']

            rating_cnt_data=rating_cnt_data.head(10)

            # Bar chart
            rating_cnt_fig = px.bar(
                rating_cnt_data,
                x='model',
                y='Total_ratings',
                title='Top 10 Models by Number of User Ratings'
            )

            # Layout tweaks
            rating_cnt_fig.update_layout(
                xaxis_title=None,
                yaxis_title=None

            )

            rating_cnt_fig.update_traces(marker_color='#e85993')

            st.plotly_chart(rating_cnt_fig, use_container_width=True)

        
    col3,col4,col5 =st.columns([0.5,1,0.5])

    with col4:
        # Step 1 – remove duplicate (model, review_count) pairs
        unique_pairs = df.drop_duplicates(subset=['model', 'review_count'])

        # Step 2 – aggregate total reviews per model
        review_data = (
            unique_pairs
            .groupby('model', as_index=False)['review_count']
            .sum()
            .sort_values('review_count', ascending=False)
        )

        plot_data = review_data.head(10) 

        # Step 4 – plot
        review_fig = px.bar(
            plot_data,
            x='model',
            y='review_count',
            title='Top 10 Models by Number of User Reviews'
        )

        # Step 5 – layout polish
        review_fig.update_layout(
            xaxis_title=None,
            yaxis_title=None
        )

        review_fig.update_traces(marker_color='#bbb929')

        st.plotly_chart(review_fig, use_container_width=True)


# ==== TAB 4: Price Analysis ====
with tabs[3]:
    st.subheader("💸Understanding Pricing Patterns and User Perception")

    price_hist_fig = px.histogram(
    df,
    x='price',
    nbins=30,  # adjust number of bins for granularity
    title='Number of Models by Price Range',
)

    price_hist_fig.update_layout(
        xaxis_title=None,
        yaxis_title=None
    )

    st.plotly_chart(price_hist_fig, use_container_width=True)

    # Scatterplot: Price vs Rating

    # 📉 Reduce duplicates by keeping only the most-reviewed variant per model
    df_top = (
        df.sort_values('review_count', ascending=False)
          .groupby('model', as_index=False)
          .first()  # keeps the first (i.e., most reviewed) row per model
    )



#### Scatterplot
    scatter_fig = px.scatter(
        df_top,
        x="price",
        y="rating",
        color="series",          # 🟦 Legend by Samsung series
        hover_data=["model", "generation", "chipset", "price", "rating"],
        title="Price vs Rating"
    )

    # 💡 Uniformly increase point size
    scatter_fig.update_traces(
        marker=dict(
            size=14,            # ⬆️ globally larger bubble size (default ~6)
            opacity=0.8,
            line=dict(width=0.6, color='white')
        ),
        selector=dict(mode='markers')
    )

    scatter_fig.update_layout(
        xaxis_title="Price",
        yaxis_title="User Rating",
        legend_title_text="Phone Series",
        height=650,              # ⬆️ Chart height
    )

    st.plotly_chart(scatter_fig, use_container_width=True)





# ==== TAB 5: Recommendation System ====
with tabs[4]:
    st.subheader("📱 Identifying the Best-Value Samsung Devices Across Different Series")
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #1e293b, #334155);
            color: #f8fafc;
            padding: 14px 18px;
            border-radius: 12px;
            font-size: 15px;
            line-height: 1.6;
            box-shadow: 0 2px 6px rgba(0,0,0,0.25);
        ">
        <b>✨ How the recommendations are made</b><br>
        Each model is ranked using a <b>weighted scoring system</b> across key specs:
        <ul style="margin-top:6px;">
            <li>💰 <b>Price </b> – strongest influence; lower price increases value.</li>
            <li>⚡ <b>CPU Performance </b> – Second most influential spec; faster cpu saves time from opening apps to browsing.</li>
            <li>🔋 <b>Battery Capacity </b> – longer battery life gives an edge.</li>
            <li>💾 <b>RAM & Storage </b> – higher capacity improves usability and can handle large apps and files.</li>
            <li>📷 <b>Camera Setup </b> – additional lenses allows customers to capture versatile shots and manage any scenarios.</li>
            <li>⭐ <b>User Ratings & Reviews </b> – reflects customer satisfaction and trust.</li>
        </ul>
        </div>
        <br><br>
        """,
        unsafe_allow_html=True
    )


    # --- Step 1: Select Tier ---
    available_tiers = sorted(df['tier'].dropna().unique())
    selected_tier = st.selectbox(
        "Select a Tier:",
        available_tiers,
        index=None    )

    if selected_tier is None:
        st.info("Please select a Tier to continue.")
        st.stop()

    # --- Step 2: Select Series ---
    available_series = sorted(
        df[df['tier'] == selected_tier]['series'].dropna().unique()
    )

    selected_series = st.selectbox(
        "Select a Series:",
        available_series,
        index=None    )

    if selected_series is None:
        st.info("Please select a Series to continue.")
        st.stop()

    # --- Step 3: Safe to filter now ---
    selected_data = df[
        (df['tier'] == selected_tier) &
        (df['series'] == selected_series)
    ].copy()


    if selected_data.empty:
        st.warning("No data available for this Tier and Series.")
    else:
        # --- Normalization function (used for both within-group and global) ---
        def normalize(x, higher_is_better=True):
            if x.nunique() <= 1:
                return 1.0
            return ((x - x.min()) / (x.max() - x.min())) if higher_is_better else ((x.max() - x) / (x.max() - x.min()))

        # --- Within-generation normalization ---
        selected_data['rating_norm']        = selected_data.groupby('generation')['rating'].transform(normalize, True)
        selected_data['rating_count_norm']  = selected_data.groupby('generation')['rating_count'].transform(normalize, True)
        selected_data['price_norm']         = selected_data.groupby('generation')['price'].transform(normalize, False)
        selected_data['battery_norm']       = selected_data.groupby('generation')['battery_mah'].transform(normalize, True)
        selected_data['rearcam_norm']       = selected_data.groupby('generation')['rearcam_count'].transform(normalize, True)
        selected_data['ram_norm']           = selected_data.groupby('generation')['ram_gb'].transform(normalize, True)
        selected_data['storage_norm']       = selected_data.groupby('generation')['storage_gb'].transform(normalize, True)
        selected_data['cpu_norm']           = selected_data.groupby('generation')['cpu_score'].transform(normalize, True)

        # --- Recommendation score ---
        selected_data['recommendation_score (Out of 100)'] = (
            0.05 * selected_data['rating_norm'] +
            0.05 * selected_data['rating_count_norm'] +
            0.35 * selected_data['price_norm'] +
            0.10 * selected_data['battery_norm'] +
            0.05 * selected_data['rearcam_norm'] +
            0.10 * selected_data['ram_norm'] +
            0.10 * selected_data['storage_norm'] +
            0.20 * selected_data['cpu_norm']
        )

        selected_data = selected_data.dropna(subset=['recommendation_score (Out of 100)'])

        # --- Best per generation ---
        best_per_generation = (
            selected_data.loc[
                selected_data.groupby('generation')['recommendation_score (Out of 100)'].idxmax()
            ].copy()
        )

        # --- Drop normalization columns ---
        best_variant = best_per_generation.drop(
            columns=[c for c in best_per_generation.columns if c.endswith('_norm') or c in ['color']],
            errors='ignore'
        )

        # --- Global normalization ---
        for col, higher in [
            ('rating', True), ('rating_count', True), ('price', False),
            ('battery_mah', True), ('rearcam_count', True),
            ('ram_gb', True), ('storage_gb', True), ('cpu_score',True)
        ]:
            best_variant[f'{col}_norm'] = normalize(best_variant[col], higher)

        # --- Final score ---
        best_variant['recommendation_score (Out of 100)'] = (
            0.05 * best_variant['rating_norm'] +
            0.05 * best_variant['rating_count_norm'] +
            0.35 * best_variant['price_norm'] +
            0.10 * best_variant['battery_mah_norm'] +
            0.05 * best_variant['rearcam_count_norm'] +
            0.10 * best_variant['ram_gb_norm'] +
            0.10 * best_variant['storage_gb_norm'] +
            0.20 * best_variant['cpu_score_norm']
        )


        # --- Top 2 ---
        top2_models = (
            best_variant.sort_values(by='recommendation_score (Out of 100)', ascending=False)
            .head(2)
            .reset_index(drop=True)
        )

        medals = ['🥇 Gold', '🥈 Silver']
        top2_models['Rank'] = medals[:len(top2_models)]

        # --- Display ---
        st.success(f"🏆 Top 2 Recommended {selected_series}-Series Models in {selected_tier} Tier")
        st.dataframe(
            top2_models[
                ['Rank', 'model', 'chipset', 'price', 'rating', 'rating_count',
                 'battery_mah', 'rearcam_count', 'ram_gb', 'storage_gb', 'recommendation_score (Out of 100)']
            ],
            use_container_width=True
        )



st.markdown("---")

st.caption("Built with Streamlit • Built by Joyal K Noble.")
