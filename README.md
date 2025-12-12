# Capstone Project – CPE312 Introduction to Data Science    
 กลุ่มห้าในใจเธอ

---

##  Table of Contents
1. [Project Overview](#project-overview)  
2. [Objective](#objective)  
3. [Dataset](#dataset)  
4. [Tools & Technologies](#tools--technologies)  
5. [Data Preparation](#data-preparation)  
6. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)  
7. [Modeling](#modeling)  
8. [Model Evaluation](#model-evaluation)  
9. [Key Findings](#key-findings)  
10. [Conclusion](#conclusion)  
11. [Project Members](#project-members)

---

## 1.  Project Overview

โปรเจกต์นี้มีวัตถุประสงค์เพื่อ **วิเคราะห์แนวโน้มและพยากรณ์อัตราความยากจน** ในพื้นที่ต่าง ๆ  
โดยใช้ข้อมูล SDG Indicator 1.2.1 (Below Poverty Line) ของ San Diego County  
พร้อมทั้งสร้างโมเดล Machine Learning และ Time-Series เพื่อทำนายพื้นที่เสี่ยง

โครงงานนี้ครอบคลุมตั้งแต่ Data Wrangling, EDA, Forecasting

---

## 2.  Objective

###  เป้าหมายของโปรเจกต์นี้ ได้แก่
- วิเคราะห์แนวโน้มความยากจนรายปี  
- ระบุปีแรกที่ความยากจนเพิ่มขึ้นในแต่ละพื้นที่  
- ทำ Forecast อัตราความยากจนในอนาคต  
- สร้างรายงานเชิงวิเคราะห์เพื่อช่วยในการวางแผนด้านสังคมและเศรษฐกิจ  

---

## 3.  Dataset
- แหล่งข้อมูล: **SANDAG Open Data Portal**  
- ชุดข้อมูล: **Below Poverty Line (SDG Indicator 1.2.1)**  
- จำนวนตัวแปรสำคัญ:
  - Year  
  - Geography Name  
  - Poverty Percent  
  - Poverty Estimate
  - Population Estimate

ข้อมูลนี้ถูกใช้เพื่อวิเคราะห์และสร้างโมเดลแบบ Time-Series และ Classification

---

## 4.  Tools & Technologies

| หมวด | เครื่องมือ |
|------|------------|
| Programming | Python (Pandas, NumPy) |
| Data Visualization | Matplotlib, Seaborn |
| Time-Series Model | Facebook Prophet, ARIMA |
| Machine Learning | Random Forest, XGBoost |
| Metrics | MAE, RMSE, Confusion Matrix, Accuracy, F1-score |
| Version Control | Git, GitHub |

---

## 5.  Data Preparation

ขั้นตอนเตรียมข้อมูล (Data Wrangling):
- ลบ missing values  
- แปลงตัวเลขให้เป็น numeric  
- เลือกเฉพาะคอลัมน์ที่จำเป็น  
- จัดเรียงข้อมูลตามปี  
- Group ข้อมูลตามพื้นที่  
- แปลงข้อมูลให้อยู่ในรูป Time-Series (ds, y) เพื่อใช้กับ Prophet  

นี่คือขั้นตอนสำคัญที่ช่วยให้โมเดลทำงานถูกต้องและลด error ในภายหลัง

---

## 6.  Exploratory Data Analysis (EDA)

EDA ของโปรเจกต์นี้ประกอบด้วย:

###  1. Line Chart  
แสดง **Poverty Yearly Trend** ของแต่ละพื้นที่  

###  2. Bar Chart  
เปรียบเทียบอัตราความยากจนระหว่างพื้นที่  

###  3. Histogram  
ดูการกระจายตัวและตรวจสอบความเบ้ของข้อมูล  

###  4. Box Plot  
ใช้ตรวจสอบ Outliers  

###  5. Correlation Heatmap  
ดูความสัมพันธ์ระหว่างตัวแปร เช่น Population vs Poverty Estimate

EDA ทำให้เข้าใจข้อมูลเชิงลึกและช่วยเลือกโมเดลที่เหมาะสมที่สุด

---

## 7.  Modeling

โปรเจกต์นี้สร้างโมเดลทั้งหมด 4 ประเภท:

###  1. Prophet
โมเดลที่ให้ความแม่นยำสูงที่สุดสำหรับ Forecast ความยากจนในอนาคต  

###  2. ARIMA  
โมเดล Series แบบดั้งเดิม ใช้เปรียบเทียบคุณภาพกับ Prophet  

###  3. Machine Learning Regression  
- Random Forest  
- XGBoost  

---

## 8.  Model Evaluation

###  Error Metrics (Regression)
โมเดลถูกประเมินด้วย:
- MAE  
- RMSE  

ผลลัพธ์: **Prophet มีค่าความผิดพลาดต่ำที่สุด → โมเดลดีที่สุด**

---

## 9.  Key Findings

- แนวโน้มความยากจนมีการเปลี่ยนแปลงตามพื้นที่และช่วงปี  
- Prophet เหมาะสำหรับการพยากรณ์ข้อมูลแบบ Time-Series  
- Machine Learning ไม่เหมาะกับชุดข้อมูลขนาดเล็กที่มีลักษณะเป็น Time-Series  
- พบ Outliers และ Distribution เบ้ ซึ่งส่งผลต่อการเรียนรู้ของโมเดล  
- ได้เรียนรู้การใช้ Git เพื่อจัดการเวอร์ชันโค้ดและแยกงานเป็นไฟล์อย่างเป็นระบบ  

---

## 10.  Project Members

- สุภาวรรณ โพธิ์ใต้  
- ธนาภา มนตราลักษณ์  
- ทีฑายุ จันทิพย์  
- วรไกรกาญจน์ บินการญจน์  
- นัสเราะห์ ดือเร๊ะ  
