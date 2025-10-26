# Laporan Proyek Machine Learning 

## Employee Attrition Prediction

**Natasha Virna Arthilita** — *5003231020*  
**Violita Inggar Ariana** — *5003231021*  
**Fahmadina Sophia** — *5003231122*

## 🗂️ Daftar Isi

1. [Domain Proyek: Manajemen Sumber Daya](#1-domain-proyek-manajamen-sumber-daya)  
   - [Referensi](#referensi)  
2. [Business Understanding](#2-business-understanding)  
   - [Problem Statements](#problem-statements)  
   - [Goals](#goals)  
   - [Solution Statements](#solution-statements)  
   - [Project Benefits](#project-benefits)  
3. [Data Understanding](#3-data-understanding)  
   - [Sumber Data](#sumber-data)  
   - [Deskripsi Fitur](#deskripsi-fitur)  
   - [Penjelasan Kontekstual Fitur](#penjelasan-kontekstual-fitur)  
   - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)  
     - [Missing Value & Outliers](#missing-value--outliers)  
     - [Univariate Analysis](#univariate-analysis)  
     - [Multivariate Analysis](#multivariate-analysis)  
     - [Kesimpulan EDA](#kesimpulan-eda)  
4. [Data Preparation](#4-data-preparation)  
   - [Label Encoding dengan Mapping pada Fitur Target](#label-encoding-dengan-mapping-pada-fitur-target)  
   - [Splitting Dataset](#splitting-dataset)  
   - [Feature Engineering, Data Cleaning and Preprocessing](#feature-engineering-data-cleaning-and-preprocessing)  
5. [Model Training, Comparison, Selection and Tuning](#5-model-training-comparison-selection-and-tuning)  
   - [Model Selection](#model-selection)  
   - [Feature Selection](#feature-selection)  
   - [Hyperparameter Tuning](#hyperparameter-tuning)  
6. [Model Testing and Evaluation](#6-model-testing-and-evaluation)  
   - [Data Test Predict](#data-test-predict)  
   - [Best Model Evaluation](#best-model-evaluation)  
     - [Classification Report](#classification-report)  
     - [Metode Evaluasi Lanjutan](#metode-evaluasi-lanjutan)  
     - [Confusion Matrix](#confusion-matrix)  
     - [Plot ROC-AUC Curve](#plot-roc-auc-curve)  
     - [Plot PR-AUC Curve](#plot-pr-auc-curve)  
7. [Save Best Model](#7-save-best-model)  
8. [Model Interpretation](#8-model-interpretation)  
   - [Interpretation with SHAP Values](#interpretation-with-shap-values)  
   - [Feature Importance](#feature-importance)  
9. [Financial Result](#9-financial-result)  
10. [Conclusions](#10-conclusions)  
    - [Ringkasan Proyek](#ringkasan-proyek)  
    - [Hasil dan Evaluasi Model](#hasil-dan-evaluasi-model)  
    - [Penanganan Ketidakseimbangan Data](#penanganan-ketidakseimbangan-data)  
    - [Interpretasi dan Validasi Model](#interpretasi-dan-validasi-model)  
    - [Estimasi Nilai Finansial](#estimasi-nilai-finansial)  
    - [Langkah Selanjutnya](#langkah-selanjutnya)

## Domain Proyek : Manajemen Sumber Daya
![Employee Attrition](https://github.com/natashavirnaa/employee-attrition-prediction/blob/main/image/employe-attrition.png?raw=true)
Di dunia bisnis modern, pengelolaan sumber daya manusia menjadi faktor kunci bagi keberlanjutan dan daya saing perusahaan [1]. Salah satu tantangan utama dalam bidang ini adalah tingginya tingkat attrition atau employee turnover, yaitu kondisi ketika karyawan meninggalkan perusahaan baik secara sukarela maupun tidak. Tingkat turnover yang tinggi dapat menimbulkan biaya besar bagi perusahaan, mengganggu produktivitas, serta berdampak negatif terhadap moral tim [2].
Untuk mengatasinya, banyak organisasi kini mengandalkan pendekatan berbasis data guna memahami faktor-faktor yang memengaruhi keputusan karyawan dalam bertahan atau keluar dari perusahaan [3]. Dengan memanfaatkan analisis prediktif, perusahaan dapat mengidentifikasi pola perilaku karyawan, mendeteksi risiko turnover lebih dini, serta merancang strategi retensi yang lebih efektif.
Dataset yang digunakan dalam proyek ini mencakup berbagai aspek profil karyawan, lingkungan kerja, dan status karyawan (bertahan atau keluar). Variabel-variabel tersebut meliputi faktor demografis, jabatan, jam lembur, tingkat kepuasan kerja, dan beban kerja, yang dapat digunakan untuk membangun model prediktif employee attrition dan mengeksplorasi fitur-fitur yang paling berpengaruh terhadap keputusan karyawan [4].

---

**Referensi**
<div style="font-size: 10px; line-height: 1.6; margin-left: 10px;">
[1] Dessler, G. (2020). <i>Human Resource Management</i> (16th ed.). Pearson Education.<br>
[2] Cascio, W. F. (2015). <i>Managing Human Resources: Productivity, Quality of Work Life, Profits</i> (10th ed.). McGraw-Hill Education.<br>
[3] Nguyen, T., & Duong, M. (2021). “Predicting Employee Turnover Using Machine Learning Techniques.” <i>Journal of Human Resource Analytics,</i> 5(3), 45–57.<br>
[4] IBM HR Analytics Employee Attrition & Performance Dataset (2019). <i>Kaggle.</i><br>
<a href="https://www.kaggle.com/datasets/pavansubhashh/ibm-hr-analytics-attrition-dataset">https://www.kaggle.com/datasets/pavansubhashh/ibm-hr-analytics-attrition-dataset</a>
</div>

---

## Business Understanding
### Problem Statemnents
Dalam dunia kerja modern, setiap perusahaan berusaha mempertahankan karyawan berprestasi untuk menjaga stabilitas operasional dan daya saing. Namun, tingginya tingkat employee attrition atau turnover menjadi tantangan serius bagi manajemen sumber daya manusia. Fenomena ini dapat menyebabkan peningkatan biaya rekrutmen dan pelatihan, penurunan produktivitas, serta menurunnya moral kerja tim secara keseluruhan.

Berdasarkan hal tersebut, pernyataan masalah yang diangkat dalam proyek ini adalah sebagai berikut: 
1. **Pernyataan Masalah 1:** Bagaimana mengidentifikasi faktor-faktor utama yang memengaruhi keputusan karyawan untuk keluar dari perusahaan?  
2. **Pernyataan Masalah 2:** Bagaimana membangun model prediktif yang mampu memperkirakan kemungkinan seorang karyawan akan keluar dengan akurasi yang tinggi?  
3. **Pernyataan Masalah 3:** Bagaimana memanfaatkan hasil analisis tersebut untuk merumuskan strategi retensi karyawan yang efektif dan berbasis data?

### Goals
Untuk menjawab pernyataan masalah di atas, tujuan proyek ini adalah:  
1. **Tujuan 1:** Melakukan eksplorasi dan analisis terhadap data karyawan untuk menemukan pola dan variabel yang berkorelasi dengan perilaku attrition.
2. **Tujuan 2:** Membangun model prediktif berbasis machine learning untuk menghitung probabilitas seorang karyawan akan keluar dari perusahaan.
3. **Tujuan 3:** Menyediakan insight dan rekomendasi strategis yang dapat membantu manajemen dalam meningkatkan retensi karyawan dan menciptakan lingkungan kerja yang lebih produktif.

### Solution Statements
Untuk mencapai tujuan tersebut, solusi yang akan diterapkan meliputi:  
- **Eksperimen dengan Berbagai Algoritma Klasifikasi**
  Membangun dan membandingkan performa beberapa algoritma seperti:
  - Logistic Regression
  - Random Forest
  - XGBoost / LightGBM
- **Optimasi Model dengan Hyperparameter Tuning**
  Menggunakan pendekatan seperti Grid Search atau Optuna (Bayesian Optimization) untuk menemukan konfigurasi parameter terbaik.
- **Evaluasi Model dengan Metrik yang Relevan**
  Menerapkan metrik evaluasi seperti:
  - Accuracy — untuk mengukur kinerja keseluruhan model.
  - Precision, Recall, dan F1-Score — untuk menilai performa pada kelas minoritas (attrition).
  - ROC-AUC — untuk mengevaluasi kemampuan model dalam membedakan karyawan yang bertahan dan keluar.
  - Confusion Matrix — untuk memvisualisasikan hasil prediksi model.
- **Analisis Fitur Dan visualisasi**
  Menyajikan visualisasi seperti feature importance dan correlation heatmap untuk menginterpretasikan fitur-fitur utama yang berkontribusi pada analisis.

### Project Benefit
Dengan implementasi solusi ini, manfaat yang diharapkan antara lain:
- **Efisiensi Biaya SDM:** Mengurangi biaya rekrutmen dan pelatihan akibat tingginya turnover.
- **Peningkatan Retensi Karyawan:** Mengidentifikasi dan menindaklanjuti karyawan dengan risiko tinggi untuk keluar.
- **Pengambilan Keputusan Berbasis Data:** Membantu manajemen dalam menyusun strategi retensi dan kebijakan HR yang lebih tepat sasaran.
- **Perbaikan Lingkungan Kerja:** Memberikan insight terhadap faktor-faktor yang menyebabkan ketidakpuasan dan menurunkan motivasi kerja.
- **Produktivitas yang Lebih Stabil:** Menjaga kontinuitas tim dan mengurangi dampak negatif dari pergantian karyawan.

---
## Data Understanding
### Sumber Data
Dataset yang digunakan dalam proyek ini diperoleh dari situs Kaggle. Dataset ini berisi data profil dan lingkungan kerja karyawan yang digunakan untuk memprediksi kemungkinan attrition atau keluar dari perusahaan.

Dataset terdiri dari dua berkas utama, yaitu train.csv dengan 1.173 data latih dan test.csv dengan 295 data uji. Setiap data memuat informasi mengenai berbagai variabel yang berpotensi memengaruhi keputusan karyawan untuk tetap bertahan atau keluar.

Secara keseluruhan, terdapat **35 variabel**, termasuk satu variabel target yaitu Attrition. Variabel ini menunjukkan apakah seorang karyawan keluar (1) atau tetap bekerja (0). Sebagian kecil karyawan termasuk dalam kategori attrition, sehingga terdapat ketidakseimbangan kelas *(class imbalance)* yang menjadi tantangan dalam proses pelatihan model prediktif.  

## Deskripsi Fitur
| Nama Fitur | Deskripsi | Tipe Data |
|-------------|------------|-----------|
| id | ID unik karyawan untuk identifikasi. | `object` |
| Age | Usia karyawan. | `int64` |
| BusinessTravel | Frekuensi perjalanan dinas karyawan. | `object` |
| DailyRate | Gaji harian. | `int64` |
| Department | Departemen tempat karyawan bekerja. | `object` |
| DistanceFromHome | Jarak tempat tinggal karyawan ke kantor. | `int64` |
| Education | Tingkat pendidikan (1 = Below College, 2 = College, 3 = Bachelor, 4 = Master, 5 = Doctor). | `int64` |
| EducationField | Bidang studi terakhir karyawan. | `object` |
| EmployeeCount | Jumlah karyawan. | `int64` |
| EmployeeNumber | Nomor unik karyawan dalam sistem HR. | `int64` |
| EnvironmentSatisfaction | Tingkat kepuasan terhadap lingkungan kerja (1 = Low, 2 = Medium, 3 = High, 4 = Very High). | `int64` |
| Gender | Jenis kelamin karyawan. | `object` |
| HourlyRate | Upah per jam. | `int64` |
| JobInvolvement | Tingkat keterlibatan pekerjaan (1 = Low, 2 = Medium, 3 = High, 4 = Very High). | `int64` |
| JobLevel | Level jabatan karyawan. | `int64` |
| JobRole | Posisi/jabatan spesifik karyawan. | `object` |
| JobSatisfaction | Tingkat kepuasan pekerjaan (1 = Low, 2 = Medium, 3 = High, 4 = Very High). | `int64` |
| MaritalStatus | Status pernikahan karyawan. | `object` |
| MonthlyIncome | Gaji bulanan karyawan. | `int64` |
| MonthlyRate | Tarif bulanan karyawan. | `int64` |
| NumCompaniesWorked | Jumlah perusahaan tempat karyawan pernah bekerja sebelumnya. | `int64` |
| Over18 | Status usia di atas 18 tahun. | `object` |
| OverTime | Status lembur karyawan. | `object` |
| PercentSalaryHike | Persentase kenaikan gaji tahunan terakhir. | `int64` |
| PerformanceRating | Penilaian kinerja terakhir. | `int64` |
| RelationshipSatisfaction | Tingkat kepuasan terhadap hubungan kerja (1 = Low, 2 = Good, 3 = Excellent, 4 = Outstanding). | `int64` |
| StandardHours | Jam kerja standar (selalu 80 dalam dataset). | `int64` |
| StockOptionLevel | Level kepemilikan saham perusahaan. | `int64` |
| TotalWorkingYears | Total tahun pengalaman kerja. | `int64` |
| TrainingTimesLastYear | Jumlah pelatihan yang diikuti selama setahun terakhir. | `int64` |
| WorkLifeBalance | Tingkat keseimbangan kerja-hidup (1 = Bad, 2 = Good, 3 = Better, 4 = Best). | `int64` |
| YearsAtCompany | Total tahun bekerja di perusahaan saat ini. | `int64` |
| YearsInCurrentRole | Total tahun di posisi/jabatan saat ini. | `int64` |
| YearsSinceLastPromotion | Tahun sejak promosi terakhir. | `int64` |
| YearsWithCurrManager | Tahun bekerja dengan manajer saat ini. | `int64` |
| Attrition | Target: apakah karyawan keluar dari perusahaan (1 = Yes, 0 = No). | `object` |

### [Exploratory Data Analysis] - Statistika Deskriptif
| Fitur | count | mean | std | min | 25% | 50% | 75% | max |
|-------|--------|------|------|------|------|------|------|------|
| Age | 1176.0 | 37.00 | 9.18 | 18.0 | 30.0 | 36.0 | 43.0 | 60.0 |
| DailyRate | 1176.0 | 803.99 | 401.34 | 103.0 | 467.75 | 799.5 | 1157.0 | 1499.0 |
| DistanceFromHome | 1176.0 | 9.37 | 8.18 | 1.0 | 2.0 | 7.0 | 14.0 | 29.0 |
| Education | 1176.0 | 2.91 | 1.03 | 1.0 | 2.0 | 3.0 | 4.0 | 5.0 |
| EmployeeCount | 1176.0 | 1.00 | 0.00 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| EmployeeNumber | 1176.0 | 1015.83 | 599.66 | 1.0 | 487.75 | 1004.5 | 1547.25 | 2062.0 |
| EnvironmentSatisfaction | 1176.0 | 2.72 | 1.09 | 1.0 | 2.0 | 3.0 | 4.0 | 4.0 |
| HourlyRate | 1176.0 | 65.50 | 20.37 | 30.0 | 48.0 | 66.0 | 83.0 | 100.0 |
| JobInvolvement | 1176.0 | 2.74 | 0.70 | 1.0 | 2.0 | 3.0 | 3.0 | 4.0 |
| JobLevel | 1176.0 | 2.08 | 1.09 | 1.0 | 1.0 | 2.0 | 3.0 | 5.0 |
| JobSatisfaction | 1176.0 | 2.72 | 1.11 | 1.0 | 2.0 | 3.0 | 4.0 | 4.0 |
| MonthlyIncome | 1176.0 | 6544.02 | 4653.74 | 1009.0 | 2948.0 | 5004.5 | 8420.5 | 19973.0 |
| MonthlyRate | 1176.0 | 14390.24 | 7192.83 | 2094.0 | 8051.0 | 14373.0 | 20770.75 | 26999.0 |
| NumCompaniesWorked | 1176.0 | 2.69 | 2.49 | 0.0 | 1.0 | 2.0 | 4.0 | 9.0 |
| PercentSalaryHike | 1176.0 | 15.24 | 3.68 | 11.0 | 12.0 | 14.0 | 18.0 | 25.0 |
| PerformanceRating | 1176.0 | 3.16 | 0.36 | 3.0 | 3.0 | 3.0 | 3.0 | 4.0 |
| RelationshipSatisfaction | 1176.0 | 2.74 | 1.09 | 1.0 | 2.0 | 3.0 | 4.0 | 4.0 |
| StandardHours | 1176.0 | 80.00 | 0.00 | 80.0 | 80.0 | 80.0 | 80.0 | 80.0 |
| StockOptionLevel | 1176.0 | 0.79 | 0.85 | 0.0 | 0.0 | 1.0 | 1.0 | 3.0 |
| TotalWorkingYears | 1176.0 | 11.36 | 7.80 | 0.0 | 6.0 | 10.0 | 15.0 | 40.0 |
| TrainingTimesLastYear | 1176.0 | 2.76 | 1.26 | 0.0 | 2.0 | 3.0 | 3.0 | 6.0 |
| WorkLifeBalance | 1176.0 | 2.76 | 0.72 | 1.0 | 2.0 | 3.0 | 3.0 | 4.0 |
| YearsAtCompany | 1176.0 | 7.05 | 6.09 | 0.0 | 3.0 | 5.0 | 10.0 | 37.0 |
| YearsInCurrentRole | 1176.0 | 4.23 | 3.57 | 0.0 | 2.0 | 3.0 | 7.0 | 17.0 |
| YearsSinceLastPromotion | 1176.0 | 2.18 | 3.22 | 0.0 | 0.0 | 1.0 | 3.0 | 15.0 |
| YearsWithCurrManager | 1176.0 | 4.20 | 3.56 | 0.0 | 2.0 | 3.0 | 7.0 | 17.0 |
| Attrition | 1176.0 | 0.16 | 0.37 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |

Dataset ini mencakup **1.176 karyawan** dengan rata-rata usia **37 tahun**, yang menunjukkan dominasi **usia produktif** di lingkungan kerja. Pendapatan bulanan rata-rata mencapai **Rp6.544**, dengan variasi yang cukup besar antar karyawan. Rata-rata masa kerja karyawan di perusahaan ini sekitar **7 tahun**, sementara total pengalaman kerja mereka mencapai **11 tahun**.

Tingkat **kepuasan terhadap pekerjaan**, **lingkungan kerja**, dan **hubungan antar rekan** berada di kisaran **2,7 dari 4**, yang mengindikasikan tingkat kepuasan sedang. Sebagian besar karyawan pernah bekerja di sekitar **tiga perusahaan sebelumnya**, dengan rata-rata **kenaikan gaji tahunan sebesar 15%**.

Proporsi karyawan yang keluar dari perusahaan (**Attrition = 1**) tercatat sebesar **16%**, menunjukkan tingkat **turnover yang relatif rendah**, namun tetap **penting untuk dianalisis lebih lanjut**. Secara keseluruhan, data ini menggambarkan **profil tenaga kerja yang relatif stabil**, tetapi dengan variasi dalam **pendapatan**, **masa kerja**, dan **kepuasan kerja** yang berpotensi memengaruhi keputusan karyawan untuk **bertahan atau meninggalkan perusahaan**.

### Perbandingan Rata-Rata Variabel Berdasarkan Status Attrition  
| Fitur | Attrition = 0 | Attrition = 1 |
|-------|----------------|----------------|
| Age | 37.74 | 33.13 |
| DailyRate | 816.56 | 738.74 |
| DistanceFromHome | 9.05 | 10.97 |
| Education | 2.92 | 2.84 |
| EmployeeCount | 1.00 | 1.00 |
| EmployeeNumber | 1009.80 | 1047.15 |
| EnvironmentSatisfaction | 2.77 | 2.44 |
| HourlyRate | 65.67 | 64.60 |
| JobInvolvement | 2.78 | 2.54 |
| JobLevel | 2.16 | 1.66 |
| JobSatisfaction | 2.77 | 2.46 |
| MonthlyIncome | 6871.64 | 4843.88 |
| MonthlyRate | 14321.50 | 14747.10 |
| NumCompaniesWorked | 2.64 | 2.97 |
| PercentSalaryHike | 15.25 | 15.16 |
| PerformanceRating | 3.16 | 3.16 |
| RelationshipSatisfaction | 2.76 | 2.63 |
| StandardHours | 80.00 | 80.00 |
| StockOptionLevel | 0.84 | 0.52 |
| TotalWorkingYears | 12.00 | 8.06 |
| TrainingTimesLastYear | 2.78 | 2.65 |
| WorkLifeBalance | 2.78 | 2.63 |
| YearsAtCompany | 7.46 | 4.90 |
| YearsInCurrentRole | 4.49 | 2.87 |
| YearsSinceLastPromotion | 2.24 | 1.87 |
| YearsWithCurrManager | 4.46 | 2.83 |

Karyawan yang **keluar dari perusahaan (Attrition = 1)** umumnya memiliki **usia lebih muda**, dengan rata-rata sekitar **33 tahun**, dibandingkan karyawan yang bertahan (**37 tahun**). Mereka juga memiliki **pendapatan bulanan** dan **tingkat jabatan** yang lebih rendah, serta **masa kerja yang lebih singkat** di perusahaan. Selain itu, **tingkat keterlibatan kerja**, **kepuasan terhadap pekerjaan**, dan **kepuasan terhadap lingkungan kerja** cenderung lebih rendah pada kelompok ini.

Karyawan yang keluar juga memiliki **total pengalaman kerja** dan **masa bekerja bersama manajer** yang lebih pendek, serta **kesempatan promosi** yang lebih sedikit. Meskipun perbedaan dalam **kenaikan gaji tahunan** dan **work-life balance** relatif kecil, pola ini menunjukkan bahwa **faktor kepuasan kerja** dan **peluang pengembangan karier** menjadi pendorong utama dalam keputusan karyawan untuk **meninggalkan perusahaan**.

### [Exploratory Data Analysis] - Menangani Missing Value dan Outliers
Dalam tahap awal pembersihan data, dilakukan pengecekan terhadap duplikasi data dan *missing value*. Hasilnya menunjukkan bahwa tidak terdapat duplikasi data maupun *missing value* di seluruh kolom fitur maupun target. Hal ini mengindikasikan bahwa dataset sudah lengkap dan tidak memerlukan teknik imputasi lebih lanjut.
| Nama Fitur | Data Train |
|-------------|------------|
| id | 0 |
| Age | 0 |
| BusinessTravel | 0 |
| DailyRate | 0 |
| Department | 0 |
| DistanceFromHome | 0 |
| Education | 0 |
| EducationField | 0 |
| EmployeeCount | 0 |
| EmployeeNumber | 0 |
| EnvironmentSatisfaction | 0 |
| Gender | 0 |
| HourlyRate | 0 |
| JobInvolvement | 0 |
| JobLevel | 0 |
| JobRole | 0 |
| JobSatisfaction | 0 |
| MaritalStatus | 0 |
| MonthlyIncome | 0 |
| MonthlyRate | 0 |
| NumCompaniesWorked | 0 |
| Over18 | 0 |
| OverTime | 0 |
| PercentSalaryHike | 0 |
| PerformanceRating | 0 |
| RelationshipSatisfaction | 0 |
| StandardHours | 0 |
| StockOptionLevel | 0 |
| TotalWorkingYears | 0 |
| TrainingTimesLastYear | 0 |
| WorkLifeBalance | 0 |
| YearsAtCompany | 0 |
| YearsInCurrentRole | 0 |
| YearsSinceLastPromotion | 0 |
| YearsWithCurrManager | 0 |
| Attrition | 0 |

Selanjutnya, dilakukan **deteksi *outlier*** menggunakan metode **Interquartile Range (IQR)** untuk setiap fitur numerik.  
Hasil analisis menunjukkan bahwa beberapa variabel memiliki jumlah *outlier* yang cukup signifikan.  
Fitur **Attrition (190 *outlier*)**, **PerformanceRating (185 *outlier*)**, dan **TrainingTimesLastYear (174 *outlier*)** merupakan variabel dengan jumlah *outlier* terbanyak.  
Kondisi ini mengindikasikan adanya variasi ekstrem dalam tingkat performa, frekuensi pelatihan, serta status keluar atau bertahannya karyawan.  

Selain itu, fitur seperti **MonthlyIncome (86 *outlier*)**, **YearsSinceLastPromotion (85 *outlier*)**, dan **StockOptionLevel (66 *outlier*)** juga menunjukkan keberadaan *outlier* yang cukup tinggi.Hal ini mencerminkan adanya perbedaan besar dalam pendapatan, frekuensi promosi, dan kepemilikan saham antar karyawan.  

Beberapa variabel lain seperti **TotalWorkingYears (52 *outlier*)** dan **YearsAtCompany (52 *outlier*)** juga menunjukkan variasi signifikan dalam lama pengalaman kerja dan masa kerja di perusahaan. Sementara itu, fitur seperti **JobSatisfaction**, **JobLevel**, **JobInvolvement**, serta berbagai variabel demografis dan lingkungan kerja lainnya **tidak memiliki *outlier*** sama sekali, dimana hal ini menandakan distribusi nilai yang relatif seragam dan stabil di antara karyawan.

| Fitur | Jumlah Outlier |
|-------|----------------|
| Attrition | 190 |
| PerformanceRating | 185 |
| TrainingTimesLastYear | 174 |
| MonthlyIncome | 86 |
| YearsSinceLastPromotion | 85 |
| StockOptionLevel | 66 |
| TotalWorkingYears | 52 |
| YearsAtCompany | 52 |
| NumCompaniesWorked | 36 |
| YearsInCurrentRole | 16 |
| YearsWithCurrManager | 10 |
| JobSatisfaction | 0 |
| JobLevel | 0 |
| JobInvolvement | 0 |
| HourlyRate | 0 |
| EnvironmentSatisfaction | 0 |
| EmployeeNumber | 0 |
| EmployeeCount | 0 |
| Education | 0 |
| DistanceFromHome | 0 |
| DailyRate | 0 |
| Age | 0 |
| PercentSalaryHike | 0 |
| StandardHours | 0 |
| RelationshipSatisfaction | 0 |
| MonthlyRate | 0 |
| WorkLifeBalance | 0 |

### [Exploratory Data Analysis] - Univariate Analysis
#### Grafik 1 : Distribusi Kategori Employee Attrition
![Distribusi Kategori Employee-Attrition](https://github.com/natashavirnaa/employee-attrition-prediction/blob/main/image/EDA%201%20-%20Distribusi%20Kategori%20Employee-Attrition.png?raw=true)

Distribusi variabel **Attrition** menunjukkan bahwa sebagian besar karyawan **tidak keluar dari perusahaan (sekitar 84%)**, sedangkan hanya sekitar **16% yang mengalami attrition (keluar)**.  
Kondisi ini **menandakan adanya ketidakseimbangan kelas** yang cukup besar antara karyawan yang bertahan dan yang keluar, yang dapat **mempengaruhi performa model prediksi**.  

Untuk mengatasi hal tersebut, dilakukan beberapa strategi berikut:
1. **Stratified Train-Test Split**  
   Pembagian data dilakukan secara *terstratifikasi* agar proporsi antara karyawan yang keluar dan tidak keluar tetap konsisten di data pelatihan dan pengujian.  
2. **Penyesuaian Bobot Kelas (*Class Weight Adjustment*)**  
   Model diberikan bobot lebih besar pada kelas minoritas (**Attrition = 1**) agar kesalahan prediksi terhadap karyawan yang keluar tidak diabaikan.  
3. **Evaluasi dengan Metrik Sensitif terhadap Ketidakseimbangan**  
   Penggunaan metrik seperti **Recall**, *F1-Score*, dan **ROC-AUC** difokuskan agar model tidak hanya akurat pada kelas mayoritas, tetapi juga mampu mengenali pola karyawan yang berpotensi keluar.

#### Grafik 2 : Distribusi Fitur Numerik
![Distribusi Fitur Numerik](https://github.com/natashavirnaa/employee-attrition-prediction/blob/main/image/EDA%202%20-%20Distribusi%20Fitur%20Numerik.png?raw=true)

- Sebagian besar karyawan berada pada **rentang usia 30–40 tahun**, dengan distribusi yang relatif normal.  
  Pendapatan bulanan (*MonthlyIncome*) dan total tahun bekerja (*TotalWorkingYears*) menunjukkan **distribusi miring ke kanan**, menandakan hanya sebagian kecil karyawan dengan gaji dan masa kerja tinggi.  
- Variabel seperti *DailyRate*, *MonthlyRate*, dan *HourlyRate* tampak menyebar merata, sementara *PerformanceRating* dan *StandardHours* cenderung **konstan** dengan sedikit variasi.  
  Fitur seperti *EnvironmentSatisfaction*, *JobSatisfaction*, dan *RelationshipSatisfaction* memiliki distribusi yang **hampir seragam antar kategori**, menunjukkan persebaran tingkat kepuasan yang beragam di antara karyawan.  
- Beberapa fitur seperti *YearsAtCompany*, *YearsInCurrentRole*, dan *YearsSinceLastPromotion* memiliki **bentuk distribusi menurun tajam**, menandakan banyak karyawan yang baru bekerja atau belum lama dipromosikan.  
  Temuan ini memberikan gambaran umum bahwa struktur tenaga kerja didominasi oleh karyawan dengan masa kerja menengah, tingkat kepuasan yang bervariasi, serta distribusi pendapatan yang tidak merata.

#### Grafik 3 : Distribusi Variabel Kategorik
![Distribusi Variabel Kategorik](https://github.com/natashavirnaa/employee-attrition-prediction/blob/main/image/EDA%203%20-%20Distribusi%20Variabel%20Kategorik.png?raw=true)
Mayoritas karyawan bekerja di **Departemen Research & Development**, diikuti oleh **Sales**, sedangkan **Human Resources** menjadi departemen dengan jumlah karyawan paling sedikit.  
Sebagian besar karyawan **jarang melakukan perjalanan dinas** (*Travel_Rarely*) dan memiliki latar belakang pendidikan di bidang **Life Sciences** atau **Medical**.  

Dari sisi demografi, karyawan laki-laki tercatat sedikit lebih banyak dibandingkan perempuan, dan sebagian besar **berstatus menikah**. Dalam kategori pekerjaan (*JobRole*), posisi **Sales Executive** dan **Research Scientist** merupakan yang paling dominan. Selain itu, hampir seluruh karyawan berusia di atas **18 tahun (*Over18 = Y*)**, dan sebagian besar **tidak bekerja lembur (*OverTime = No*)**.  

Secara keseluruhan, temuan ini menggambarkan bahwa **struktur tenaga kerja perusahaan cukup beragam**, meskipun terdapat **ketidakseimbangan antar kategori** tertentu — seperti dominasi departemen **R&D** serta rendahnya proporsi karyawan yang melakukan lembur.  
Kondisi ini dapat menjadi faktor yang **mempengaruhi hasil analisis attrition** pada tahap selanjutnya.  

### [Exploratory Data Analysis] - Multivariate Analysis
![Korelasi Fitur Numerik](https://github.com/natashavirnaa/employee-attrition-prediction/blob/main/image/EDA%204%20-%20Korelasi%20Fitur%20Numerik.png?raw=true)
- **OverTime** memiliki **korelasi positif tertinggi**, yang menunjukkan bahwa karyawan yang sering lembur cenderung memiliki risiko lebih tinggi untuk keluar dari perusahaan.  
- Fitur seperti **MonthlyIncome**, **TotalWorkingYears**, dan **JobLevel** menunjukkan **korelasi negatif** terhadap *attrition* — semakin tinggi pendapatan, pengalaman kerja, dan level jabatan, semakin kecil kemungkinan karyawan untuk keluar.  
- **Age** juga memiliki korelasi negatif, yang mengindikasikan bahwa karyawan yang lebih tua cenderung bertahan lebih lama dibandingkan karyawan yang lebih muda.  
- Sebaliknya, **DistanceFromHome** memiliki **korelasi positif kecil**, yang berarti jarak tempat tinggal yang lebih jauh dari kantor sedikit meningkatkan risiko *attrition*.  
- Selain itu, korelasi antar beberapa fitur seperti **MonthlyIncome** dan **JobLevel** cukup tinggi, menunjukkan adanya potensi **redundansi antar variabel** yang berhubungan dengan pendapatan dan posisi kerja.  

![BoxPlot Distribusi Variabel Numerik](https://github.com/natashavirnaa/employee-attrition-prediction/blob/main/image/EDA%205%20-%20BoxPlot%20Distribusi%20Variabel%20Numerik.png?raw=true)
Berdasarkan hasil boxplot di atas, persebaran dari fitur numerik terhadap variabel *attrition* dapat dijelaskan sebagai berikut:  
**Karyawan yang resign (*Attrition = 1*) cenderung memiliki:**  
- **Usia lebih muda**, mayoritas berada pada rentang **25–35 tahun**, dibandingkan dengan karyawan yang bertahan yang umumnya berusia di atas 35 tahun.  
- **Pendapatan lebih rendah**, baik dari sisi *MonthlyIncome*, *HourlyRate*, maupun *DailyRate*.  
- **Level jabatan** (*JobLevel*) serta **total tahun bekerja** (*TotalWorkingYears*) yang lebih sedikit.  
- **Masa kerja dengan manajer saat ini** (*YearsWithCurrManager*) dan **masa kerja di perusahaan** (*YearsAtCompany*) yang lebih pendek — menunjukkan tingkat loyalitas dan pengalaman yang lebih rendah.  
- **Frekuensi promosi** (*YearsSinceLastPromotion*) yang lebih sedikit, mengindikasikan kemungkinan adanya ketidakpuasan karier.  
- **Tingkat kepuasan kerja dan lingkungan** (*JobSatisfaction*, *EnvironmentSatisfaction*, *RelationshipSatisfaction*) yang umumnya lebih rendah dibandingkan dengan karyawan yang bertahan.  

**Sementara itu, karyawan yang tidak mengalami attrition (*Attrition = 0*) cenderung:**  
- **Berpenghasilan lebih tinggi** dengan variasi pendapatan yang lebih besar.  
- **Memiliki masa kerja yang lebih panjang**, baik secara total maupun pada posisi saat ini.  
- **Lebih sering mendapatkan kesempatan promosi** serta memiliki **tingkat kepuasan kerja yang lebih tinggi**.

## Kesimpulan EDA
Berdasarkan hasil eksplorasi data yang telah dilakukan, dapat disimpulkan bahwa:
1. **Distribusi Attrition Tidak Seimbang**  
   Sebagian besar karyawan (**84%**) tercatat **tidak keluar dari perusahaan**, sementara **16%** mengalami *attrition* (keluar). Ketidakseimbangan ini menunjukkan bahwa data bersifat *imbalanced*, sehingga perlu perhatian khusus dalam pemodelan, seperti penyesuaian bobot kelas atau pemilihan metrik evaluasi yang sensitif terhadap ketidakseimbangan.
2. **Profil Umum Karyawan**  
   Mayoritas karyawan berusia **30–40 tahun** dengan **masa kerja sekitar 7 tahun** dan total pengalaman kerja **11 tahun**. Pendapatan bulanan rata-rata berada pada kisaran **Rp6.544**, menunjukkan variasi pendapatan yang cukup besar antar individu.
3. **Faktor-Faktor yang Berkaitan dengan Attrition**  
   Karyawan yang keluar (*Attrition = 1*) cenderung memiliki:
   - Usia yang lebih muda (25–35 tahun).  
   - Pendapatan dan level jabatan yang lebih rendah.  
   - Masa kerja yang lebih singkat, baik di perusahaan maupun dengan manajer saat ini.  
   - Frekuensi promosi yang lebih sedikit.  
   - Tingkat kepuasan kerja dan lingkungan yang lebih rendah.  
   Sebaliknya, karyawan yang bertahan cenderung memiliki pendapatan dan jabatan yang lebih tinggi, serta tingkat kepuasan yang lebih baik.
4. **Korelasi antar Variabel**  
   Beberapa variabel seperti **MonthlyIncome**, **TotalWorkingYears**, dan **JobLevel** memiliki korelasi negatif terhadap *attrition*, menunjukkan bahwa peningkatan faktor-faktor tersebut dapat menurunkan kemungkinan karyawan keluar.  
   Sementara itu, **OverTime** memiliki korelasi positif paling kuat, menandakan bahwa karyawan yang sering lembur lebih berisiko untuk meninggalkan perusahaan.
5. **Gambaran Departemen dan Demografi**  
   Sebagian besar karyawan bekerja di **Departemen Research & Development**, disusul oleh **Sales** dan **Human Resources**.  
   Karyawan laki-laki sedikit lebih banyak dibandingkan perempuan, dan mayoritas berstatus menikah. Posisi dominan dipegang oleh **Sales Executive** dan **Research Scientist**. Sebagian besar karyawan jarang melakukan perjalanan dinas (*Travel_Rarely*).

Secara keseluruhan, analisis EDA menunjukkan bahwa **attrition di perusahaan dipengaruhi oleh faktor usia, pendapatan, masa kerja, tingkat promosi, dan kepuasan kerja**.  
Karyawan dengan **penghasilan rendah, masa kerja singkat, serta kepuasan kerja yang rendah** lebih berisiko untuk keluar.  
Temuan ini dapat menjadi dasar penting dalam proses **pemodelan prediktif attrition**, sekaligus memberikan wawasan bagi manajemen dalam menyusun strategi **retensi karyawan yang lebih efektif dan berbasis data**.

---

## Data Preparation 
### 1. Label Encoding dengan Mapping pada Fitur Target 
Proses encoding dilakukan secara manual untuk fitur target attrition.  
Mapping digunakan sebagai berikut:
| Kategori Attrition | Keterangan                 | Label |
|---------------------|----------------------------|:------:|
| No                  | Tidak resign (bertahan)    | 0 |
| Yes                 | Resign (keluar dari kerja) | 1 |

### 2. Splitting Dataset
Menetapkan `stratify = y` sehingga fungsi `train_test_split` memastikan bahwa proses pemisahan mempertahankan persentase yang sama dari setiap kelas target di set *train* dan *validation*.  
Dataset yang digunakan dalam analisis ini terdiri dari data pelatihan (*train*) dan data validasi (*validation*) dengan rincian sebagai berikut:
- **Ukuran data fitur (train):** 940 observasi dengan 51 fitur.  
- **Ukuran data target (train):** 940 observasi.  
- **Ukuran data fitur (validation):** 236 observasi dengan 51 fitur.  
- **Ukuran data target (validation):** 236 observasi.  

---
**Proporsi Kelas pada Variabel Target**
Distribusi proporsi kelas pada variabel target `Attrition` untuk masing-masing dataset adalah sebagai berikut:
**Data Pelatihan (Train):**
- Kelas 0 (*Tidak Attrition / Tetap Bekerja*): **83,80%**  
- Kelas 1 (*Attrition / Keluar Perusahaan*): **16,20%**  
**Data Validasi (Validation):**
- Kelas 0 (*Tidak Attrition / Tetap Bekerja*): **83,90%**  
- Kelas 1 (*Attrition / Keluar Perusahaan*): **16,10%**

---
Distribusi kelas yang relatif konsisten antara data pelatihan dan validasi menunjukkan bahwa proses pembagian data dengan parameter `stratify = y` telah berhasil **mempertahankan keseimbangan proporsi kelas target**.  
Dengan demikian, model dapat dilatih dan dievaluasi secara konsisten terhadap fenomena *attrition* tanpa bias distribusi data.

### 3. Feature Engineering, Data Cleaning and Preprocessing
#### **Pre-Processing**
- **Fitur Numerik**  
  Tidak dilakukan transformasi atau normalisasi.  
  Model berbasis pohon keputusan (seperti **XGBoost**, **LightGBM**, atau **Random Forest**) tidak sensitif terhadap skala data, sehingga **feature scaling tidak diperlukan**.  
- **Fitur Kategorikal**  
  Fitur: *Education*, *EnvironmentSatisfaction*, *JobInvolvement*, *JobSatisfaction*, *PerformanceRating*, *RelationshipSatisfaction*, *WorkLifeBalance*.  
  Beberapa variabel yang disebutkan di atas menggunakan **Ordinal Encoding** untuk mempertahankan urutan nilai yang mencerminkan tingkat atau level sebenarnya dari setiap kategori.  
- **Nominal Features**
  Fitur: *BusinessTravel*, *Department*, *EducationField*, *JobRole*, *MaritalStatus*  
Diterapkan **Target Encoding**, karena:
   - *One-Hot Encoding* dapat menyebabkan *sparse matrix* dan meningkatkan dimensi data secara signifikan.
   - *Target Encoding* lebih efisien untuk model *tree-based*, serta mampu menangkap proporsi rata-rata target (*Attrition*) di setiap kategori.
- **Fitur Biner (*Gender*, *OverTime*)**  
  Diterapkan **One-Hot Encoding**, karena hanya memiliki dua kategori.  
  Transformasi ini akan menghasilkan variabel biner (0/1) tanpa meningkatkan dimensi data secara signifikan.
- **Fitur Konstan**  
  Fitur: *EmployeeCount*, *Over18*, dan *StandardHours*.  
  Fitur-fitur ini memiliki **nilai konstan pada semua entri**, sehingga dihapus karena **tidak memberikan informasi tambahan terhadap model**.

#### Feature Engineering
Untuk memperkuat kemampuan model dalam menangkap hubungan non-linear serta pola tersembunyi dalam data karyawan, dilakukan pembuatan sejumlah **fitur turunan**.  
Fitur-fitur ini membantu meningkatkan interpretabilitas model dan memberikan konteks bisnis yang lebih kaya terhadap faktor penyebab *attrition*.  
**🔹 Core Ratio Features**
Fitur-fitur rasio ini menggambarkan efisiensi, stabilitas, dan pengalaman kerja karyawan:
```text
1. ExperienceRatio = YearsAtCompany / (TotalWorkingYears + 1)
   → Mengukur proporsi pengalaman kerja yang dihabiskan di perusahaan saat ini.
2. CurrentRoleRatio = YearsInCurrentRole / (YearsAtCompany + 1)
   → Menggambarkan stabilitas posisi jabatan selama masa kerja.
3. JobHoppingRate = NumCompaniesWorked / (TotalWorkingYears + 1)
   → Mengindikasikan kecenderungan berpindah-pindah perusahaan.
4. ManagerStability = YearsWithCurrManager / (YearsAtCompany + 1)
   → Menunjukkan durasi dan stabilitas hubungan dengan manajer langsung.
```

**🔹 Binary Indicators**
Dibuat sejumlah indikator biner untuk merepresentasikan kondisi risiko tinggi terhadap attrition:
```text
1. IsYoung = 1 jika Age < 30  
   → Mengindikasikan kelompok usia muda dengan mobilitas tinggi.  
2. TimeWithoutPromotion = 1 jika YearsSinceLastPromotion > 3   
   → Menandakan stagnasi karier dalam jangka waktu lama.  
3. LongCommute = 1 jika DistanceFromHome > 15  
   → Mengindikasikan jarak rumah jauh dari kantor yang berpotensi menurunkan loyalitas.  
4. PoorWorkLife = 1 jika WorkLifeBalance ≤ 2  
   → Menunjukkan keseimbangan hidup dan pekerjaan yang buruk.  
5. OverTime_Binary = 1 jika OverTime = Yes  
   → Mengindikasikan frekuensi lembur yang tinggi dan potensi burnout.  
6. LowJobLevel = 1 jika JobLevel ≤ 1  
   → Mengindikasikan posisi jabatan rendah.
```

**🔹 Satisfaction & Career Dynamics** 
Beberapa fitur gabungan dibuat untuk menangkap hubungan antar aspek produktivitas, kepuasan, dan kinerja:
```text
1. AvgSatisfaction = mean(EnvironmentSatisfaction, JobSatisfaction, 
                          RelationshipSatisfaction, WorkLifeBalance)
   → Menggambarkan tingkat kepuasan kerja secara keseluruhan.  
2. TotalPerformance = PerformanceRating + AvgSatisfaction
   → Menunjukkan keseimbangan antara kinerja dan kepuasan kerja.  
3. OverallStability = ManagerStability + CurrentRoleRatio
   → Mewakili tingkat kestabilan posisi dan hubungan kerja.
```

**🔹 Extended Analytical Features** 
Untuk memperkaya informasi prediktif, ditambahkan kombinasi fitur yang mencerminkan **produktivitas, pendapatan, dan kinerja karyawan**.    
Fitur-fitur ini membantu model memahami hubungan antara faktor ekonomi dan performa kerja yang dapat berkontribusi terhadap *attrition*.  
```text
1. IncomePerYear = MonthlyIncome / (TotalWorkingYears + 1)
   → Menggambarkan rata-rata pendapatan per tahun pengalaman kerja.  
2. Income_JobLevel = MonthlyIncome × JobLevel
   → Mengindikasikan hubungan antara pendapatan dan posisi jabatan.  
3. Age_Experience = Age × TotalWorkingYears
   → Menggambarkan korelasi antara usia dan pengalaman kerja.  
4. Satisfaction_Performance = AvgSatisfaction × PerformanceRating
   → Mengukur keseimbangan antara kepuasan kerja dan kinerja aktual.  
5. Seniority = JobLevel × TotalWorkingYears
   → Mewakili tingkat senioritas karyawan secara umum.  
6. Promotion_Rate = YearsAtCompany / (YearsSinceLastPromotion + 1)
   → Menggambarkan frekuensi promosi relatif terhadap masa kerja.  
```

## Model Training, Comparison, Selection and Tuning
### 1. Model Selection
Pada tahap pengembangan model, dilakukan proses pemilihan beberapa algoritma klasifikasi untuk menentukan model terbaik dalam memprediksi *Employee Attrition*. Proses ini dilakukan dengan membandingkan performa dari berbagai model baik linear maupun berbasis pohon keputusan (*tree-based models*) guna menemukan keseimbangan antara akurasi, stabilitas, dan kemampuan generalisasi.

Tiga model utama yang dievaluasi secara mendalam adalah **Logistic Regression**, **Random Forest**, dan **LightGBM**, dengan masing-masing mewakili pendekatan berbeda dalam *machine learning**. Pemilihan ketiganya didasarkan pada kemampuan mereka menangani data tabular, interpretabilitas hasil, serta efisiensi pelatihan.

**Logistic Regression (L1 & L2 Regularized)**
```python
from sklearn.linear_model import LogisticRegression

log_model_l2 = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000, random_state=42)
log_model_l2.fit(X_train_scaled, y_train
```
Model **Logistic Regression** digunakan sebagai baseline linear untuk mengukur performa dasar prediksi.  
Model ini bekerja dengan mengestimasi probabilitas suatu observasi masuk ke kelas tertentu melalui fungsi logit (sigmoid). Untuk menghindari *overfitting*, diterapkan regularisasi **L1 (Lasso)** dan **L2 (Ridge)** yang berfungsi menekan kompleksitas model dengan mengurangi bobot koefisien yang kurang signifikan.  

**Random Forest**
```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight='balanced'
)
rf_model.fit(X_train_scaled, y_train)
```
**Random Forest** merupakan algoritma *ensemble learning* berbasis *bagging* yang menggabungkan banyak pohon keputusan independen.  
Setiap pohon dilatih menggunakan subset acak dari data dan fitur, sehingga variasi antar pohon tinggi dan hasil agregasi menjadi lebih stabil.  
Pendekatan ini efektif untuk mengurangi *variance* dan risiko *overfitting* pada model pohon tunggal seperti Decision Tree.

**LightGBM**
```python
from lightgbm import LGBMClassifier

lgbm_model = LGBMClassifier(
    boosting_type='gbdt',
    n_estimators=500,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=31,
    random_state=42
)
lgbm_model.fit(X_train_scaled, y_train)
```
**LightGBM (Light Gradient Boosting Machine)** adalah algoritma *boosting* berbasis pohon yang dikembangkan untuk efisiensi tinggi dan performa superior pada dataset besar.  
Berbeda dengan pendekatan tradisional, LightGBM membangun pohon secara *leaf-wise* — memilih daun dengan *gain* terbesar pada setiap langkah — yang menghasilkan model lebih presisi dengan waktu pelatihan cepat.

Ketiga model ini digunakan dengan pengaturan parameter awal sebagai percobaan dasar untuk melihat performa awal dan kestabilan hasil.  
Evaluasi dilakukan menggunakan teknik **stratified k-fold cross validation** untuk memastikan bahwa setiap lipatan (*fold*) data memiliki proporsi kelas target yang seimbang.  
Hal ini sangat penting pada kasus *Employee Attrition* karena data cenderung tidak seimbang antara karyawan yang bertahan dan yang keluar.
- Pada langkah ini, kinerja setiap model dibandingkan menggunakan *stratified k-fold cross validation*, di
  mana setiap model dilatih dan dievaluasi pada lipatan berbeda untuk memperoleh skor ROC-AUC rata-rata.
  Pendekatan ini menjaga keseimbangan kelas dan membantu menilai performa generalisasi model.  
- *k-fold cross validation* sendiri merupakan teknik umum dalam *machine learning* untuk menilai kemampuan model secara objektif.Dataset dibagi menjadi **K subset**, di mana **K-1** subset digunakan untuk pelatihan dan 1 subset sisanya untuk pengujian. Proses ini diulang sebanyak *K kali* agar setiap bagian data digunakan baik untuk pelatihan maupun pengujian. Teknik ini membantu memperkirakan kemampuan generalisasi model dengan mengurangi risiko *overfitting* dan memberikan metrik performa yang lebih stabil dan dapat dipercaya.
- Tujuan dari tahap ini adalah menentukan model terbaik yang akan digunakan pada proses **feature selection**, **hyperparameter tuning**, dan **evaluasi akhir**.  
Model dengan performa validasi rata-rata **ROC-AUC tertinggi** dan *gap* pelatihan–validasi yang kecil dianggap sebagai model yang paling ideal.
  
**Tabel Perbandingan Performa Model**

| **Model**                 | **ROC-AUC (Val)** | **Gap**     | **Catatan**                    |
|----------------------------|:----------------:|:-----------:|--------------------------------|
| Logistic L2 (Tuned)        | 0.8321           | +0.0175     | Stabil dan interpretatif       |
| Logistic L1 (Tuned)        | 0.7275           | -0.0354     | Kurang stabil                  |
| Random Forest (Optimized)  | 0.7919           | +0.1269     | Overfit ringan                 |
| Gradient Boosting          | 0.8008           | +0.1898     | Overfit moderat                |
| LightGBM (Tuned)           | 0.7872           | +0.2086     | Performa tinggi, efisien       |
| CatBoost                   | 0.8121           | +0.1550     | Konsisten                      |
| AdaBoost                   | 0.7834           | +0.0729     | Baik untuk baseline            |
| Bagging (Logistic Base)    | 0.8345           | +0.0020     | Sangat stabil                  |

Berdasarkan hasil validasi dan efisiensi komputasi, model LightGBM dipilih untuk proses *feature selection, hyperparameter tuning*, dan evaluasi akhir.
Model ini menunjukkan performa ROC-AUC yang tinggi dan stabil, serta memiliki kemampuan generalisasi yang baik meskipun menunjukkan sedikit indikasi *overfitting*.

**LightGBM**juga mendukung interpretasi lebih lanjut melalui *feature importance* dan analisis nilai SHAP, sehingga memudahkan eksplorasi faktor-faktor utama yang memengaruhi tingkat Employee Attrition.

### 2. Feature Selection

- Langkah seleksi fitur sangat penting untuk meningkatkan kemampuan generalisasi model sekaligus menyederhanakan kompleksitasnya, sehingga menambah efisiensi komputasi.  
  Mengingat dataset memiliki **51 fitur hasil rekayasa data**, penyederhanaan model tanpa kehilangan performa menjadi keuntungan yang signifikan. Dalam proyek ini digunakan pendekatan **Random Forest Feature Importance** untuk menentukan fitur paling relevan terhadap prediksi *employee attrition*.
- Metode **Random Forest Feature Importance** bekerja dengan melatih model pada seluruh fitur yang ada dan mengukur kontribusi relatif tiap fitur terhadap akurasi prediksi.  
  Semakin tinggi nilai *importance* suatu fitur, semakin besar pengaruhnya terhadap hasil prediksi model.  
  Proses ini digunakan untuk mengidentifikasi fitur dengan dampak terbesar terhadap variabel target, mengurangi fitur yang tidak relevan, serta meningkatkan interpretabilitas model.
- Berdasarkan hasil evaluasi, model menunjukkan bahwa beberapa fitur seperti **AttritionRiskScore**, **Age_Experience**, **MonthlyIncome**, dan **JobHoppingRate** memiliki pengaruh paling signifikan terhadap prediksi *attrition*.  
  Fitur-fitur ini berkaitan erat dengan risiko karyawan meninggalkan perusahaan, pengalaman kerja, serta kompensasi yang diterima.
- Visualisasi hasil seleksi ditampilkan dalam grafik *Feature Importance*, yang menunjukkan peringkat fitur berdasarkan kontribusinya terhadap model.  
  Grafik ini membantu memahami faktor-faktor utama yang mendorong perilaku *employee attrition* dan menjadi dasar untuk interpretasi model selanjutnya.
  ![Feature Importance](https://github.com/natashavirnaa/employee-attrition-prediction/blob/main/image/feature_importance.png?raw=true)  
  
### 3. Hyperparameter Tunning
- Dilakukan *hyperparameter tuning* pada model **LightGBM** dan **XGBoost** menggunakan metode **RandomizedSearchCV**.  
  Pendekatan ini memungkinkan pencarian parameter terbaik secara efisien dengan memilih kombinasi acak dari ruang parameter yang telah ditentukan, sehingga lebih cepat dibanding *Grid Search* tanpa mengorbankan kualitas hasil.  
- Berbeda dengan *Grid Search* yang menguji seluruh kombinasi parameter secara menyeluruh dan memakan waktu lama, *Randomized Search* hanya mengevaluasi sebagian kecil kombinasi secara acak.  
  Metode ini lebih efisien dalam kasus dataset berukuran besar seperti ini, terutama karena parameter seperti *learning rate*, *max depth*, dan *number of estimators* saling berinteraksi dan memengaruhi hasil model secara kompleks.  
- Setiap kombinasi parameter diuji menggunakan teknik **Stratified K-Fold Cross Validation** dengan nilai `k=3`, untuk memastikan distribusi kelas target tetap seimbang pada setiap *fold* dan menghindari risiko *overfitting* akibat tuning yang berlebihan terhadap data train.
- Selama proses tuning, model tetap mempertimbangkan keseimbangan kelas dengan menggunakan **class_weight='balanced'**.  
  Hyperparameter ini memungkinkan model memberikan bobot lebih besar pada kelas minoritas (*Attrition = 1*), sehingga model dapat mempelajari pola yang relevan tanpa bias terhadap kelas mayoritas.  
- Proses *hyperparameter tuning* difokuskan pada penyesuaian parameter penting seperti:
  - `num_leaves`, `max_depth` → mengontrol kompleksitas model.  
  - `learning_rate` → mengatur kecepatan pembelajaran model.  
  - `n_estimators` → menentukan jumlah pohon yang digunakan.  
  - `subsample` dan `colsample_bytree` → menentukan proporsi data dan fitur yang digunakan pada setiap iterasi untuk mencegah *overfitting*.  
  - `reg_alpha` dan `reg_lambda` → pengaturan regularisasi untuk menjaga keseimbangan bias-varians.
- Hasil tuning menunjukkan bahwa model **LightGBM** memiliki performa terbaik dengan skor validasi rata-rata **ROC-AUC sebesar 0.8220**, sedikit lebih unggul dibandingkan **XGBoost (ROC-AUC = 0.8217)**.  
  LightGBM dipilih sebagai model akhir karena memiliki efisiensi pelatihan yang lebih tinggi dan stabilitas hasil yang baik pada validasi silang.
- Tahap *hyperparameter tuning* merupakan langkah penyempurnaan akhir sebelum evaluasi model.  
  Meskipun tuning berperan penting dalam peningkatan performa, kontribusi paling signifikan terhadap keberhasilan model tetap berasal dari kualitas proses *feature engineering* dan *data preprocessing* sebelumnya.

```python
final_best_params = {
    'objective': 'binary',
    'metric': 'roc_auc',
    'n_estimators': 652,
    'learning_rate': 0.0285,
    'num_leaves': 36,
    'max_depth': 4,
    'min_child_samples': 46,
    'subsample': 0.8199,
    'colsample_bytree': 0.9976,
    'reg_alpha': 0.0692,
    'reg_lambda': 4.1486,
    'class_weight': 'balanced',
    'bagging_freq': 1,
    'verbosity': -1,
    'random_state': 42,
    'n_jobs': -1
}
```
Hasil tuning menunjukkan bahwa LightGBM memberikan performa validasi terbaik (ROC-AUC 0.8220),
sedangkan XGBoost menghasilkan nilai yang sangat mendekati (ROC-AUC 0.8217).
Kedua model ini kemudian digunakan untuk dibandingkan lebih lanjut pada tahap Model Testing and Evaluation

## Model Testing and Evaluation
Tahap ini bertujuan untuk membandingkan performa berbagai model yang telah melalui proses pelatihan dan tuning,  
serta menentukan model terbaik yang akan digunakan untuk prediksi akhir dan tahap deployment.  
Evaluasi dilakukan secara bertahap melalui proses **training**, **comparison**, **calibration**, **ensemble**, dan **testing**.

---

### Model Training
Beberapa model diuji dan dioptimalkan melalui proses pelatihan terpisah, baik untuk model linear maupun berbasis pohon keputusan.  
Berikut adalah ringkasan performa pelatihan yang mencakup nilai *training AUC*, *validation AUC*, *cross-validation AUC*, dan *generalization gap*:
| Model | Train | Val | CV | Gap | Keterangan |
|:------|:------:|:------:|:------:|:------:|:-------------|
| Logistic L2 (Tuned) | 0.8496 | 0.8321 | 0.8144 | 0.0175 | ✅ Good |
| Logistic L1 (Tuned) | 0.6921 | 0.7275 | 0.6387 | -0.0354 | ✅ Good |
| Random Forest (Optimized) | 0.9187 | 0.7919 | 0.7858 | 0.1269 | ⚠️ Overfit |
| Extra Trees (Optimized) | 0.9472 | 0.8215 | 0.8005 | 0.1256 | ⚠️ Overfit |
| Gradient Boosting (Optimized) | 0.9905 | 0.8008 | 0.8238 | 0.1898 | ⚠️ Overfit |
| XGBoost (Optimized) | 0.8963 | 0.8091 | 0.8156 | 0.0871 | ⚠️ Warn |
| Bagging (Logistic Base) | 0.8365 | 0.8345 | 0.8110 | 0.0020 | ✅ Excellent |

Dari hasil di atas, model **Bagging (Logistic Base)** menunjukkan *generalization gap* paling kecil (+0.0020),  
yang menandakan stabilitas tinggi dan kemampuan generalisasi terbaik terhadap data baru.

### Model Comparison
Untuk memastikan konsistensi performa, seluruh model dibandingkan kembali berdasarkan hasil validasi dan *cross-validation (CV)*.

| Model | Train | Val | CV | Gap |
|:------|:------:|:------:|:------:|:------:|
| Gradient Boosting (Optimized) | 0.9905 | 0.8007 | 0.8237 | 0.1898 |
| CatBoost (Optimized) | 0.9670 | 0.8121 | 0.8225 | 0.1549 |
| LightGBM (Tuned) | 0.9958 | 0.7872 | 0.8169 | 0.2086 |
| XGBoost (Optimized) | 0.8963 | 0.8091 | 0.8156 | 0.0871 |
| Logistic L2 (Tuned) | 0.8496 | 0.8321 | 0.8144 | 0.0175 |
| **Bagging (Logistic Base)** | **0.8365** | **0.8345** | **0.8110** | **0.0020** |

🔹 **Hasil terbaik diperoleh oleh model Bagging (Logistic Base)** dengan *Validation AUC* sebesar **0.8345** dan *CV AUC* 0.8110.  
Meskipun model berbasis pohon (seperti LightGBM atau CatBoost) memiliki performa kuat,  
model tersebut menunjukkan *gap* yang lebih besar, menandakan potensi *overfitting*.

### Model Calibration
Tahap ini dilakukan untuk memastikan bahwa probabilitas yang dihasilkan model telah terkalibrasi dengan baik.  
Tiga model dengan performa terbaik (Gradient Boosting, CatBoost, dan LightGBM) diuji kalibrasinya menggunakan metode *Platt Scaling*.

| Model | Original AUC | Calibrated AUC | Gain |
|:------|:-------------:|:----------------:|:------:|
| Gradient Boosting | 0.8008 | 0.7987 | -0.0021 |
| CatBoost | 0.8121 | 0.8043 | -0.0078 |
| LightGBM | 0.7872 | 0.8067 | +0.0195 |

📊 Hasil menunjukkan bahwa kalibrasi pada **LightGBM** meningkatkan kualitas prediksi probabilitas secara signifikan (+0.0195),  
sementara model lainnya mengalami sedikit penurunan performa.

### Model Ensemble
Tahap ini bertujuan untuk meningkatkan performa keseluruhan model serta mengurangi *variance.*
Untuk mencapai tujuan tersebut, dilakukan proses **ensemble otomatis** yang menggabungkan beberapa model terbaik, yaitu **Logistic L2 (Tuned), XGBoost (Optimized)**, dan **Bagging (Logistic Base)**

- Model yang dipilih otomatis untuk ensemble:  
  `['Logistic L2 (Tuned)', 'XGBoost (Optimized)', 'Bagging (Logistic Base)']`
- Bobot terbaik (berdasarkan validasi):  
  - Logistic L2 (Tuned): 0.2  
  - XGBoost (Optimized): 0.2  
  - Bagging (Logistic Base): 0.6  
- **Best blended AUC (Validation): 0.8325**

🧠 Proses ensemble ini bertujuan untuk meningkatkan stabilitas hasil prediksi tanpa mengorbankan interpretabilitas,  namun tidak melampaui performa individu terbaik (Bagging).

### Test Predictions
Model hasil ensemble dan model individual terbaik diuji pada data *test set*.

| Model | Mean Predicted Probability |
|:------|:---------------------------:|
| Logistic L2 (Tuned) | 0.4210 |
| XGBoost (Optimized) | 0.1570 |
| Bagging (Logistic Base) | 0.1665 |

- **Final Mean (Ensemble): 0.2155**  
- **Train Mean: 0.1616 | Diff: 0.0539**

Dari tabel di atas rata-rata probabilitas yang stabil dan selisih kecil antara *train* dan *test* menunjukkan bahwa model tidak mengalami *data drift*.

### Final Model Decision
Berdasarkan keseluruhan hasil evaluasi, model *Bagging (Logistic Base)* ditetapkan sebagai model akhir yang akan digunakan pada tahap deployment.
Pemilihan ini didasarkan pada keseimbangan antara akurasi, stabilitas performa, dan tingkat kompleksitas model yang relatif rendah.  
Pertimbangan utama:
- Validation AUC tertinggi (0.8345) dengan gap sangat kecil (+0.0020).  
- Stabilitas performa antar *fold* dan pada data *test*.  
- Kompleksitas rendah dan interpretabilitas tinggi dibanding model berbasis pohon.  

Pada tahap ini, model digunakan untuk memprediksi probabilitas karyawan berhenti bekerja (attrition) pada data *test.* Untuk memahami sebaran hasil prediksi dan memverifikasi konsistensi distribusi antara data pelatihan (train) dan pengujian (test), dilakukan visualisasi berupa histogram dan boxplot berikut.

Dari hasil visualisasi data di atas, dapat ditarik kesimpulan sebagai berikut : 
- Model menunjukkan distribusi prediksi yang stabil dan tidak mengalami pergeseran besar antara train dan test set.
- Sebagian besar prediksi berada di area probabilitas rendah hingga menengah, yang menunjukkan model cenderung konservatif dan tidak terlalu agresif dalam memprediksi karyawan sebagai berhenti.

Secara keseluruhan, *Bagging (Logistic Base)* memberikan keseimbangan optimal antara performa, stabilitas, dan interpretabilitas, sehingga layak dijadikan sebagai model akhir untuk implementasi pada sistem prediksi employee attrition di tahap produksi.

---
## Model Implementation
Model terbaik yang terpilih pada proses pemodelan adalah : 
Best Model : Bagging (Logistic Regression Base)
Validation AUC : 0,8345
AUC Gap (Train - Val) : 0,0020
Model tersimpan sebagai:
models/Bagging_Logistic_Base__v1.pkl
& bundle lengkap:
models/best_model_bundle_v1.pkl
Model terbaik yang dipilih adalah Bagging (Logistic Base) dengan validation AUC sebesar 0,8345 dengan gap yang sangat kecil yaitu +0,0020, Artinya bahwa model ini tidak overfitting dan memiliki stabilitas prediksi yang baik dalam memisahkan karyawan yang resign dan tetap bertahan. Hasil ini juga akan lebih baik dibanding model ensemble yang hanya memberikan AUC 0,8325. Model ini menggunakan class_weight, bukan SMOTE, sehingga model tetap belajar pola asli tanpa oversampling sintesis.
**Karakteristik Performa model : **
Akurasi Generalisasi tinggi : Validation AUC paling tinggi diantara model lain, yaitu (0,8345) dibandingkan dengan model yang lain.
Gap antara CV dan Validation AUC hanya sekitar 0,01 yang menunjukkan bahwa model stabil dan tidak overfit
Bagging dapat mengurangi variance sehingga prediksi churn lebih konsisten
**Pengaruh Fitur Utama (Based on Logistic Base Behavior)**
Karena model terbaik adalah Bagging dengan Basic Regression, maka arah interpretasi tetap mengikuti logika dari regresi logistik. 
Fitur bernilai positif → dapat meningkatkan peluang karyawan resign
Fitur bernilai negatif →  dapat menurunkan peluang karyawan untuk resign
Pada data submission terlihat bahwa memiliki data sebagai berikut : 
Mean : 0.2155, Std : 0.1360
Min : 0.0362, Max : 0.7110

| No | ID     | Probabilitas risiko keluar|
|----|--------|-------------------:|
| 1  | CM617  | 0.3664             |
| 2  | PJ010  | 0.1021             |
| 3  | GJ831  | 0.1582             |
| 4  | JD352  | 0.0593             |
| 5  | WZ263  | 0.4655             |
| 6  | OD346  | 0.2737             |
| 7  | GF698  | 0.1254             |
| 8  | JK198  | 0.1092             |
| 9  | SP276  | 0.0682             |
|10  | OP259  | 0.5508             |
|11  | CR707  | 0.2169             |

CV : 0.8238, Val : 0.8345, Ensemble : 0.8325
Berdasarkan hasil prediksi model, rata rata probabilitas risiko resign nya karyawan berada pada level rendah yaitu 0.2155, dengan variasi yang cukup besar (std 0.1360) dan rentang nilai dari 0.0362 hingga 0.7110. Data menunjukkan bahwa beberapa karyawan memiliki risiko cukup tinggi untuk keluar(resign) seperti OP259 sebesar 0.5508 dan WZ263 sebesar 0.4655. Pada model yang stabil dan akurat tercermin dari nilzi CV sebesar 0.8238, validation sebesar 0.8345 dan ensemble 0.8325 yang artinya bahwa model dapat dipakai sebagai dasar untuk ambil kesimpulan.


---
## Save Best Model
```python
filename = '../model/Bagging_Logistic_Base__v1.pkl'
---
## Model Interpretation
Model terbaik yang terpilih pada proses pemodelan adalah :
**Best Model:** `Bagging (Logistic Regression Base)`  
- **Validation AUC:** 0.8345  
- **AUC Gap (Train - Val):** 0.0020  
- **Model Files:**
  - Model utama: `models/Bagging_Logistic_Base__v1.pkl`
  - Bundle lengkap: `models/best_model_bundle_v1.pkl`

Model terbaik yang terpilih dalam proses pemodelan **Employee Attrition Prediction** adalah **Bagging dengan Logistic Regression Base**. Model ini mencapai **Validation AUC sebesar 0.8345** dengan **AUC gap yang sangat kecil (+0.0020)** antara data training dan validation.  
Hal ini menunjukkan bahwa model **tidak mengalami *overfitting*** dan memiliki **stabilitas prediksi yang baik** dalam membedakan antara karyawan yang **berpotensi keluar (attrition)** dan **yang bertahan**.
Hasil ini juga akan lebih baik dibanding model ensemble yang hanya memberikan AUC 0,8325. Model ini menggunakan class_weight, bukan SMOTE, sehingga model tetap belajar pola asli tanpa oversampling sintesis.

