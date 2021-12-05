# Recommendation-System-contentbased-colaborative-filtering
Make Book Recommender System with content based &amp; Colaborative Filtering 

# Laporan Proyek Machine Learning - Rizky Adhi Nugroho
---
## Project Overview 
Selama beberapa dekade terakhir, dengan munculnya banyak aplikasi seperti Youtube, Amazon, Netflix, dan banyak layanan web sejenis lainnya, sistem rekomendasi semakin banyak mengambil peran dalam kehidupan kita. sistem rekomendasi adalah algoritma yang ditujukan untuk menyarankan item yang relevan kepada pengguna.
Buku merupakan salah satu sumber informasi yang mampu membuka wawasan pembaca untuk mengetahui banyak hal. Untuk mendapatkan buku yang sesuai dengan yang diinginkan, diperlukan sebuah sistem yang dapat merekomendasikan buku untuk dibaca selanjutnya. Content based dan collaborative filtering digunakan untuk membuat suatu sistem rekomendasi yang dapat merekomendasikan buku kepada pengguna melalui nilai kesamaan antar buku. Penelitian sebelumnya pernah dilakukan oleh Muhammad Fachurrozi dan Novi Yusliani pada 2018, berikut adalah link referensinya http://generic.ilkom.unsri.ac.id/index.php/generic/article/view/81/59 

## Busines Understanding
Salah  satu  permasalahan  yang  sering  dijumpai  oleh  para  pembaca  buku  adalah menentukan buku-buku yang akan mereka baca selanjutnya. Kesulitan pembaca buku dalam menentukan buku yang akan dibaca disebabkan oleh banyaknya jumlah buku dan beragamnya jumlah buku yang ada. Solusi  untuk  permasalahan yang  dialami pembaca adalah  dengan  menerapkan  sistem rekomendasi  buku  yang  dapat  memberikan rekomendasi  buku  kepada pembaca buku.  Metode item-based dan collaborative filtering dipilih sebagai metode yang diterapkan pada sistem rekomendasi buku dikarenakan metode item-based dan collaborative filtering dapat memberikan hasil rekomendasi berdasarkan nilai kemiripan antar buku.
### Problem statements
- Berdasarkan data mengenai pengguna, bagaimana membuat sistem rekomendasi yang dipersonalisasi dengan teknik content-based filtering?
- Dengan data rating yang dimiliki, bagaimana kita dapat merekomendasikan buku lain yang mungkin disukai dan belum pernah dibaca oleh pengguna? 

### Goals
- melakukan ekplorasi data untuk menentukan variabel data apa saya yang dibutuhkan untuk membuat suatu sistem rekomendasi
- Membuat Model Sistem rekomendasi yang dapat merekomendasikan buku seakurat mungkin berdasarkan variabel data yang ada.

### Solution approach
- Menghasilkan sejumlah rekomendasi buku yang dipersonalisasi untuk pengguna dengan teknik content-based filtering
- Menghasilkan sejumlah rekomendasi buku yang sesuai dengan preferensi pengguna dan belum pernah dibaca sebelumnya dengan teknik collaborative filtering.
untuk mencapai kedua target itu, maka ditentukan 2 model untuk pembentukan sistem rekomendasi, yaitu:
  - model 1 : Content Based Filtering
content based filtering adalah metode sistem rekomendasi yang Memberikan rekomendasi berdasarkan kemiripan atribut dari item atau barang yang disukai. Pada sistem rekomendasi buku, kemiripan berdasarkan atribut yang dimiliki oleh buku seperti nama Pengarang.
  - model 2 : Collaborative Filtering
Collaborative filtering adalah metode Sistem rekomendasi yang Memberikan rekomendasi berdasarkan feedback dari user yang lain atau dari diri sendiri. Penerapan dalam rekomendasi buku yaitu pembentukan user-matrix yang berisi pereferensi dari user yang di bentuk dari data feedback yang berupa data rating dari user yang lain.

## Data Understanding
### EDA (analisis univarat)
Dataset yang dipakai adalah data dari kaggle mengenai Identitas buku (https://www.kaggle.com/arashnic/book-recommendation-dataset?select=Books.csv), data ini berisikan beberapa buku fiksi dan non fiksi yang ditulis oleh para pengarang dari beberapa negara. Dengan melakukan EDA Univariat pada data 2 file dataset, kita dapat mengetahui deskripsi dataset, yaitu dengan rincian :
- Pada file rating, terdapat 5231 baris data dan 3 kolom dengan rincian variabel :
  1. ISBN = id unik sebagai identitas dari setiap buku
  2. User-ID = id para pengguna yang memerikaan rating
  3. Book-Rating = nilai rating dari pengguna (dengan nilai 0-10)
- Pada file buku, terdapat 5427 baris data dengan 8 kolom dengan rincian variabel :
  1. ISBN = id unik sebagai identitass dari setiap buku
  2. Book-Title = judul buku
  3. Book-Author = pengarang buku
  4. Year-of-Publication = tahun terbit
  5. Publisher = penerbit
  6. image-URL-S = link gambar size kecil dari buku
  7. image-URL-M = link gambar size sedang dari buku
  8. image-URL-L = link gambar size besar dari buku
- Dalam kedua file dataset tersebut, terdapat kolom yang sama yaitu ISBN
- Terdapat 4909 data unik ISBN pada file rating, dan 5427 data unik ISBN pada file buku

2 file dataset diatas merupakan dataset raw dimana kemungkinan nantinya di tahap preprocessing dan preparation, dataset akan dirapihkan dan dibersihkan agar siap di masukkan ke model.

## Data Preparation and preprocessing
Tahapan ini diperlukan untuk mempersiapkan dataset agar bisa dimasukan dalam tahap pemodelan berikut adalah tahapan-tahapan dalam melakukan pra-pemrosesan data :
### Content-Based Filtering
#### Preprocessing (penggabungan 2 file dataset)
setelah ditemukan bahwa dalam kedua file terdapat kolom yang sama yaitu kolom ISBN, Alasan teknik penggabungan ini diperlukan karena saya akan memakai data kolom rating pada file rate untuk modeling, saya memutuskan untuk menggabungkan 2 file tersebut berdasarkan kolom ISBN. sehingga setelah penggabungan didapatkan bahwa file dataset gabungan berjumlah 5231 data dengan 10 kolom, kemudian saya memutuskan untuk melakukan drop pada kolom yang tidak terpakai yaitu kolom 'Year-Of-Publication','Publisher', 'Image-URL-s', 'Image-URL-M' dan 'Image-URL-L'. Alasan kolom kolom tersebut di drop karena tidak berpengaruh terhadap proses analisis dan pemodelan sehingga dataset sekarang tersisa 5 kolom yaitu 'User-ID', 'ISBN',	'Book-Rating', 'Book-Title', dan 'Book-Author'.

#### EDA( Cek Missing value, dan duplikasi data)
teknik ini dilakukan untuk membersihkan data yang masih mentah sebelum masuk tahap analisis, teknik ini meliputi pengecekan dan penanganan missing value, serta penanganan duplikasi data
- setelah dilakukan pengecekan, ternyata ada 4473 missing value pada kolom 'Book-Title' dan 'Book-Author'. Oleh karena itu, diperlukan drop data missing value tersebut. Alasan hal ini dilakukan karena saya tidak bisa mengidentifikasi dan mengetahui judul dan pengarang buku apa pada nomor isbn buku yang tidak memiliki data ‘Book-Title’ dan 'Book-Author' tersebut.
- setelah membersihkan missing value, tersisa 758 data. 
- Terdapat duplikasi sebanyak 183 data pada kolom ISBN yang sudah dibersihkan missing value nya, Sehingga data duplikasi ini harus di drop karena akan berpengaruh pada tahap modeling. Hal ini dilakukan karena dapat menyebabkan munculnya data yang sama sebanyak 2 kali atau lebih pada sistem rekomendasi yang nantinya dibuat. Oleh karena itu data duplikasi ini perlu dihilangkan karena sebenarnya data tersebut sudah ada pada dataset.
- Setelah dilakukan Penanganan duplikasi, terdapat 575 data final yang siap untuk dilakukan pemodelan.  

#### konversi data series menjadi list dan menentukan pasangan key-value
Alasan teknik ini dilakukan karena akan dibuat dictionary untuk persiapan pemodelan content based filtering yang membutuhkan dataframe dengan kolom yang dibutuhkan saja, yaitu data 'ISBN', 'Book-Title', dan 'Book-Author'. setelah dilakukan konversi data series menjadi list dan menentukan key value untuk data 'ISBN', 'Book-Title', dan 'Book-Author' dihasilkan dataframe sebagai berikut :

![data siap)](https://user-images.githubusercontent.com/88422709/138626862-6715c750-f866-4b95-b9a5-d331f80effa0.png)

Data tersebut Sudah Siap dimasukkan ke dalam pemodelan.

### Collaborative Filtering
Karena pada model colaborative filtering membutuhkan data rating, maka pada metode ini ada beberapa tahapan, yaitu :
- Memahami data rating yang  di miliki.
- Menyandikan (encode) fitur ‘user-id’ dan ‘ISBN’ ke dalam indeks integer.
- Memetakan ‘userID’ dan ‘ISBN’ ke dataframe yang berkaitan.
- Mengecek beberapa hal dalam data seperti jumlah user, jumlah buku, kemudian mengubah nilai rating menjadi float.
- Dilakukan pengacakan data. 
- Membagi data train dan validasi dengan komposisi 80:20.

Alasan Tahap-tahap diatas dilakukan karena kita dapat mengetahui isi dan distribusi data rating pada dataset, kita juga melakukan encoding fitur untuk merubah data menjadi tipe integer yang unik yang kemudian agar bisa dilakukan mapping antar data yang berkaitan. data rating juga diubah menjadi float karena data rating merupakan data dengan nilai skala desimal. split data dengan dilakukan pengacakan data sebelumnya dengan komposisi 80:20 agar distribusi data random dan data yang akan di masukkan ke model lebih baik.

## Modeling
### Model 1 : Content-Based Filtering
Ide dari sistem rekomendasi berbasis konten (content-based filtering) adalah merekomendasikan item yang mirip dengan item yang disukai pengguna di masa lalu. Pada Kasus ini, model ini akan merekomendasikan Pengguna Buku baru yang mirip dengan buku yang sudah dibaca dan disukai oleh pengguna tersebut berdasarkan pengarangnya.
tahapan dalam pemodelan yaitu:
#### TF - IDF Vectorizer
Pada tahap ini, kita akan membangun sistem rekomendasi sederhana berdasarkan nama pengarang dari buku. Teknik ini akan digunakan pada sistem rekomendasi untuk menemukan representasi fitur penting dari setiap kategori pengarang. Dengan menggunakan fungsi tfidfvectorizer() dari library sklearn dan melakukan fit serta transformasi ke dalam bentuk matriks, kita dapat mengidentifikasi representasi fitur penting dari setiap kategori pengarang. Kita juga telah menghasilkan matriks yang menunjukkan korelasi antara nama pengarang dengan buku.
#### Cosine Similarity
Metrik ini sering digunakan untuk mengukur kesamaan dokumen dalam analisis teks. Sebagai contoh, dalam studi kasus ini, cosine similarity digunakan untuk mengukur kesamaan nama buku dan nama pengarang. Dengan cosine similarity, kita berhasil mengidentifikasi kesamaan antara satu buku dengan buku lainnya. Berdasarkan data yang ada, matriks sebenarnya berukuran 575 buku x 575 buku (masing-masing dalam sumbu X dan Y). Artinya, kita mengidentifikasi tingkat kesamaan pada 575 nama buku. Dengan data kesamaan (similarity) buku yang diperoleh, maka kita dapat merekomendasikan daftar buku yang mirip (similar) dengan buku yang sebelumnya pernah dibaca pengguna.
#### Mendapatkan Rekomendasi
Di sini, di buat fungsi book_recommendations dengan beberapa parameter sebagai berikut:
- Nama_buku : Nama buku (index kemiripan dataframe).
- Similarity_data : Dataframe mengenai similarity yang telah kita definisikan - sebelumnya.
- Items : Nama dan fitur yang digunakan untuk mendefinisikan kemiripan, dalam hal ini adalah ‘book_name’ dan ‘pengarang’.
- k = 5 : menampilkan 5 rekomendasi yang ingin diberikan.

dengan 4 kali percobaaan dengan menggunaakan nama buku yang berbeda-beda secara random, didapatkan hasil rekomendasi sebagai berikut:
- buku 1 :

![book1](https://user-images.githubusercontent.com/88422709/138628112-ea24f929-6431-4ee0-ae97-d0081d4c52df.png)

rekomendasi buku 1:

![rbook1](https://user-images.githubusercontent.com/88422709/138628132-c3a31661-2548-4946-a4f7-087d25ace405.png)

- buku 2:

![book2](https://user-images.githubusercontent.com/88422709/138628164-3b28fd3e-358f-4b64-9250-33a67c644638.png)

rekomendasi buku 2:

![rbook2](https://user-images.githubusercontent.com/88422709/138628192-b137f974-7e3e-4260-aceb-3035d12a1e47.png)

- buku 3:

![book3](https://user-images.githubusercontent.com/88422709/138628202-c15c8680-c961-4cb3-94c0-728db7b9a9dd.png)

rekomendasi buku 3:

![rbook3](https://user-images.githubusercontent.com/88422709/138628211-b6e01280-8e7a-4d77-baa8-18a6ffe4f190.png)

- buku 4:

![book4](https://user-images.githubusercontent.com/88422709/138628204-9c509850-229a-4e5e-8d3d-310c43ce8989.png)

rekomendasi buku 4:

![rbook4](https://user-images.githubusercontent.com/88422709/138628212-c6bc7373-bd70-4c46-a139-eecec340c5f1.png)

### Model 2 : Collaborative Filtering
Collaborative filtering adalah suatu konsep dimana opini atau rating dari pengguna lain yang ada digunakan untuk memprediksi item yang mungkin disukai atau diminati oleh seorang pengguna. Dari data rating pengguna, kita akan mengidentifikasi buku-buku yang mirip dan belum pernah dibaca oleh pengguna untuk direkomendasikan. 

#### Modeling dan training data
Pada tahap ini, model menghitung skor kecocokan antara pengguna dan buku dengan teknik embedding. Pertama, kita melakukan proses embedding terhadap data user dan buku. Selanjutnya, lakukan operasi perkalian dot product antara embedding user dan buku. Selain itu, kita juga dapat menambahkan bias untuk setiap user dan buku. Skor kecocokan ditetapkan dalam skala [0,1] dengan fungsi aktivasi sigmoid. Di sini, kita membuat class RecommenderNet dengan keras Model class. Selanjutnya, di lakukan proses compile terhadap model menggunakan Binary Crossentropy untuk menghitung loss function, Adam sebagai optimizer, dan root mean squared error (RMSE) sebagai metrics evaluation. setelah model sudah siap, maka dilakukan proses training dengan 100 epoch.

#### Mendapatkan Rekomendasi Buku
Untuk mendapatkan rekomendasi buku, ambil sampel user secara acak dan definisikan variabel book_not_read untuk mendapatkan rekomendasi buku yang belum pernah dibaca. Selanjutnya, gunakan fungsi model.predict() dari library Keras dan didapatkan hasil top 10 buku rekomendasi dari user '276925' sebagai berikut:


![cresult](https://user-images.githubusercontent.com/88422709/138634268-a069ad07-1cb3-46b0-a55b-d46bf0d7d8f3.png)

## Evaluation
### Model Content Based Filtering
untuk mengevaluasi model ini, saya menggunakan metrik evaluasi precision dengan rumus dasarnya adalah:

![presisi](https://user-images.githubusercontent.com/88422709/138632176-d96e9b92-e32c-4e42-bb89-39f0fc0e9b0a.png)

dengan menggunakan rumus tersebut kita dapat menghitung hasil evaluasi dari 4 kali percobaan merekomendasikan buku yang berbeda
- buku 1 = 4 / 5 = 0.8
- buku 2 = 5 / 5 = 1.0
- buku 3 = 3 / 5 = 0.6
- buku 4 = 5 / 5 = 1.0

sehingga hasil precision = ( 0.8 + 1.0 + 0.6 + 1.0 ) / 4 = 0,85 = 85%, angka ini sudah cukup baik untuk sistem rekomendasi

### Model Collaborative Filtering
untuk mengevaluasi model ini, saya memakai Metrik evaluasi Root Mean Squared Error dengan rumus dasarnya adalah :

![RMSE](https://miro.medium.com/max/724/1*zkXt5FOfDzHNTCphWOZRSA.png)

Dengan dilakukan 100 kali epoch, didapatkan nilai rmse sebesar 0,0286 dan Val rmse sebesar 0,1850. 

![rmse](https://user-images.githubusercontent.com/88422709/138634267-3bc33e40-c0b5-48a5-a10f-5c2cd7aeb6e1.png)

Dan dilakukan visualisasi proses training dengan membuat plot metrik evaluasi 

![evalrmse](https://user-images.githubusercontent.com/88422709/138634270-4032cae2-3263-4967-96f3-4dd7943922dc.png)

dari hasil visualisasi didapatkan bahwa proses training model cukup smooth dan model konvergen pada epochs sekitar 100. Dari proses ini, kita memperoleh nilai error akhir sebesar sekitar 0.0286 dan error pada data validasi sebesar 0.1850. Nilai tersebut cukup bagus untuk sistem rekomendasi.
