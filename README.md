
# **1. Project Title / Overview**

## **Flipkart Web Scraping**

Developed a web application capable of scraping, cleaning, transforming, and analyzing Samsung 5G mobile data from Flipkart.

---

# **2. Description**

This project focuses exclusively on Samsung 5G smartphones to enable comprehensive and in-depth analysis of a single brand.
Product details are collected from **Flipkart**, while processor benchmark scores are sourced from **Antutu**.

Both datasets are cleaned, transformed, and merged.

A **Streamlit web application** is built to visualize insights using **interactive Plotly charts**, and a **recommendation system** is implemented to help customers choose the best value-for-money model.

---

# **3. Tech Stack Used**

* **Python**
* **BeautifulSoup**
* **Plotly**
* **Streamlit**

---

# **4. Highlights**

### **Web App Goals**

* Provide detailed insights into Samsung smartphones.
* Help customers choose the best value-for-money model.

---

### **Key Visuals**

* Market positioning: number of Samsung phones across tiers and series
* Top 10 models with the highest ratings and reviews
* Price distribution (Histogram)
* Price vs Rating (Scatterplot)
* Common RAM & Storage combinations

---

# **5. Challenges Faced**

Cleaning the data was the most challenging part, taking almost **75%** of the total project time.
There were numerous inconsistencies in product names, specifications, and pricing. Advanced regex-based cleaning methods were heavily used.

Missing data was another major issue. Incomplete information made meaningful analysis difficult.
To compare processor performance, benchmark scores were extracted from the **Antutu** website and merged with the Flipkart dataset. Many phones lacked processor details, which had to be added manually.

---

# **6. Web App Screenshots**

### **Data Loading Page**
![Website Traffic Dashboard](https://github.com/joyal99in/Flipkart_webscraperapp/blob/main/Data%20Loading%20Page.png)  

### **Main Page**
![Website Traffic Dashboard](https://github.com/joyal99in/Flipkart_webscraperapp/blob/main/Main%20Page.png)  

### **Product Lineup Analysis**
![Website Traffic Dashboard](https://github.com/joyal99in/Flipkart_webscraperapp/blob/main/Product%20Lineup%20Analysis.png)  

### **Hardware Analysis**
![Website Traffic Dashboard](https://github.com/joyal99in/Flipkart_webscraperapp/blob/main/Specification%20Analysis.png)  

### **Ratings & Popularity Analysis**
![Website Traffic Dashboard](https://github.com/joyal99in/Flipkart_webscraperapp/blob/main/Customer%20Satisfaction%20Analysis.png)  

### **Price Analysis**
![Website Traffic Dashboard](https://github.com/joyal99in/Flipkart_webscraperapp/blob/main/Price%20Analysis.png)  

### **Recommendation System**
![Website Traffic Dashboard](https://github.com/joyal99in/Flipkart_webscraperapp/blob/main/Recommendation%20System.png)  



