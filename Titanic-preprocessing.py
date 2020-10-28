

# Gerekli kütüphaneler yükleniyor.

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
import os
import sklearn
from sklearn.metrics import accuracy_score

# Yüklenecek büyük veri setleri için satır ve sütunların sınırsız olmasını sağlayacak kodları yazıyoruz.
pd.pandas.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
# Veri setimizde virgülden sonra 2 sıfır olması için
pd.set_option("display.float_format", lambda x: "%.2f" % x)
# şu an hangi klasörde olduğumuzu kontrol ediyoruz. Veri setinin de aynı klasörde olması gerekiyor.


titanic_dataset = pd.read_csv("4.Hafta/Dataset/titanic.csv")
df = titanic_dataset.copy()
df.head()


def load_titanic():
    data = pd.read_csv("4.Hafta/Dataset/titanic.csv")
    return data



# verisetini tanımaya başlıyoruz. Info ile değişkenleri ve tiplerini kontrol ediyoruz.
df.info()
# Yukarıda bilgisi var fakat daha özet bir şekilde kaç satır kaç sütundan oluştuğunu kontrol ediyoruz.
df.shape
# değişkenlerde eksiklerin olup olmadığını kontrol ediyoruz
df.isnull().any()
# Değişkenlerin içerisindeki değer sayılarına bakıyoruz. Ortalama ne kadar eksik olduklarını görüyoruz.
df.count()
# hangi değişkende kaç adet eksik değer olduğunu inceliyoruz.
df.isnull().sum()
a = df.isnull().sum()

# Veri setine genel bir bakış
df.describe().T
# Veri setine bakarken yüzdelik olarak detaylı inceliyoruz
df.describe([0.01, 0.1, 0.25, 0.6, 0.75, 0.9, 0.99]).T
# Age değişkenindeki eksik değerleri bir de bu şekilde inceliyoruz.
df["Age"].isnull().sum()

# Yaş değişkeninin dağılımı
sns.boxplot(x=df["Age"]).set_title("Yaş Değişkeni");
plt.show()
# Kategorik değişkenin sütun grafiği. En Çok Southampton limanından yolcu binmiş
sns.countplot(x=df["Embarked"], color="green").set_title("Embarked Grafiği");
plt.show()
# Target Analizi - Survived değişkenini inceliyoruz
df["Survived"].count()
df["Survived"].isnull().any()

# Groupby ile hayatta kalanların ve vefat edenlerin yaş ortalamasına ve max min değerlerine bakıyoruz
df.groupby("Survived").agg({"Age": ["max", "min", "mean"]})

# Survived değişkeni ile Sex değişkeninin yaşa göre kırılımlarını kontrol ediyoruz.
df.groupby(["Survived", "Sex"]).agg({"Age": ["max", "min", "mean"]})

# Sex içerisinde female geçen kelimeleri alarak df satır sayısına böldüm ve female oranını elde ettim.
df["femaleratio"] = len(df[df["Sex"].str.contains("female", na=False)]) / len(df)
##Sex içerisinde female geçMEYEN kelimeleri alarak df satır sayısına böldüm ve male oranını elde ettim.
df["maleratio"] = len(df[~df["Sex"].str.contains("female", na=False)]) / len(df)
df.head()
# deneme / Tam olarak istediğim olmadı. Sonra deneyeceğim
df.groupby(["femaleratio", "maleratio"]).agg({"Age": "mean"})
df["Sex"].value_counts()
# Hayatta kalanların ve vefat edenlerin cinsiyet ve bilet sınıflarına göre toplam rakamları inceliyoruz
df.groupby(["Survived", "Sex", "Pclass"]).agg({"Pclass": ["count"], "Age": ["mean"]})
# Yolcu segmentlerine göre bilet fiyatlarının ortalaması
df.groupby("Pclass").agg({"Fare": "mean"})

# 1) Aykırı Değer Analizi

# Veri setimize eklediğimiz sütunlar vardı. Tekrar eski haline getirmek için önceden oluşturduğumuz fonksiyonu çalıştırıyoruz.
df = load_titanic()
df.head()

# Yaş değişkeninin aykırı gözlemlerini incelemek için boxplot ile görselleştiriyoruz
sns.boxplot(x=df["Age"], color="red").set_title("Yaş Dağılımı");
plt.show()

# Age değişkeninin %25 ve %75'lik dilimlerini kontrol ediyoruz.
a = df["Age"].quantile(0.25)
b = df["Age"].quantile(0.75)
print("Age değişkeninin %25'i {}".format(a))
print("Age değişkeninin %75'i {}".format(b))
# %75 değer ile %25 değeri çıkarttığımızda IQR değerini buluyoruz
c = IQR = b - a
print("Age değişkeninin IQR değeri {}".format(c))
# IQR değerini bulduktan sonra üst sınır ve alt sınır değerlerini 1,5 ile çarparak buluyoruz
up = b + 1.5 * c
low = a - 1.5 * c
print("Age değişkeninin alt sınır değeri {}".format(low))
print("Age değişkeninin üst sınır değeri {}".format(up))


# alt sınırdan aşağıda veya üst sınırdan yukarıda var mı yok mu diye soruyoruz
df[(df["Age"] < low) | (df["Age"] > up)][["Age"]].any(axis=None)

# Alt sınrıdan küçük ve üst sınırdan büyük değerlerin sayısını inceliyoruz
df[(df["Age"] < low) | (df["Age"] > up)][["Age"]].shape[0]

# Bu sayıların hangi yaşlara ait olduklarını görüyoruz
df[(df["Age"] < low) | (df["Age"] > up)][["Age"]]

# Tüm değişkenlerin alt ve üst sınırlarını tek tek hesaplayarak bulmak yerine fonksiyon yazabiliriz


def aykiri_esikler(veri_seti, degiskenler):
    ilkceyrek = veri_seti[degiskenler].quantile(0.25)
    sonceyrek = veri_seti[degiskenler].quantile(0.75)
    ceyreklerarasi_deger = (sonceyrek - ilkceyrek)
    alt_deger = ilkceyrek - 1.5 * ceyreklerarasi_deger
    ust_deger = sonceyrek + 1.5 * ceyreklerarasi_deger
    return ust_deger, alt_deger


# oluşturduğumuz fonksiyonu çalıştırmak için parantez içine veri setini ve istediğimiz değişkeni giriyoruz
aykiri_esikler(df, "Age")

# 2 değer vereceği için bu değerleri alt, üst olarak isimlendirip istediğimizi çalıştırabiliriz
ust, alt = aykiri_esikler(df, "Age")
alt


# Bir fonksiyon yazarak değişkenlerde aykırı değer olup olmadığını hızlıca sorgulayabiliriz
def aykiri_deger_varmi(veri_seti, degiskenler):
    ust_deger, alt_deger = aykiri_esikler(veri_seti, degiskenler)
    if veri_seti[(veri_seti[degiskenler] < alt_deger) | (veri_seti[degiskenler] > ust_deger)].any(axis=None):
        print(degiskenler, ": Aykırı değer vardır!")
    else:
        print(degiskenler, ": Aykırı değer yoktur!")

aykiri_deger_varmi(df, "Age")
aykiri_deger_varmi(df, "Fare")

# aykiri değer olan değişkenleri tek fonksiyonda bulmak için oluşturduğumuz yapı
aykiri = [col for col in df.columns if len(df[col].unique()) > 10
          and df[col].dtypes != 'O'
          and col not in "PassengerId"]
aykiri

# İki fonksiyonu bir arada kullanarak değişkenleri bulup yanına da aykırı değer olup olmadığını çekiyoruz.
for col in aykiri:
    aykiri_deger_varmi(df, col)

# 1. Aykırı değer raporlamasını istiyoruz.
# 2. Aykırı değere sahip değişkenlerin box-plot'u oluşturulsun.
# 2. Bu boxplot özelliği kullanıcı tarafından biçimlendirilebilsin
# 3. Aykırı değere sahip olan değişkenlerin isimleri bir liste ile return edilsin.
# burada neden Fare değişkeni gelmiyor anlayamadım!!!


def aykiri_deger_varmi(veri_seti, degisken_adı, plot=False):
    degisken_isimleri = []

    for col in degisken_adı:
        ust_deger, alt_deger = aykiri_esikler(veri_seti, col)
        if veri_seti[(veri_seti[col] > up) | (veri_seti[col] < low)].any(axis=None):
            esik_sayisi = veri_seti[(veri_seti[col] > up) | (veri_seti[col] < low)].shape[0]

            print(col, ":", esik_sayisi)
            degisken_isimleri.append(col)
            if plot:
                sns.boxplot(x=veri_seti[col])
                plt.show()

        return degisken_isimleri


aykiri_deger_varmi(df, aykiri)
aykiri_deger_varmi(df, aykiri, plot=True)
plt.show()

# 2) Aykırı Değer Analizi

df = load_titanic()
df.shape

# Age değişkeninin alt ve üst sınırlarını tekrar hatırlıyoruz
ust, alt = aykiri_esikler(df, "Age")
print(alt)
print(ust)

# Tilda ile birlikte alt eşikten düşük olmayan ve üst eşikten yüksek olmayan verilere göz atıyoruz
df[~((df["Age"] < alt) | (df["Age"] > ust))].head()

# Tilda ile birlikte alt eşikten düşük olmayan ve üst eşikten yüksek olmayan kaç adet satır olduğunu inceliyoruz
df[~((df["Age"] < alt) | (df["Age"] > ust))].shape

# Değişkenlerdeki alt eşikten az , üst eşikten çok olan değerleri hızlıca bulmak için fonksiyon yazıyoruz

def esikleri_sil(veri_seti, degisken):
    ust_deger, alt_deger = aykiri_esikler(veri_seti, degisken)
    df_esik_harici = veri_seti[~((veri_seti[degisken] < alt) | (veri_seti[degisken] > ust))]
    return df_esik_harici


# Age değişkeninindeki eşik dışı değerleri siliyoruz.Sildiğimiz fonksiyonu df'e eşitliyoruz.
df = esikleri_sil(df, "Age")

# yeni veri setimize baktığımızda 11 eksik. 891'di 11 tane silindi
df.shape

# Aykiri fonksiyonu ile aykırı değeri olan değişkenleri buluyoruz. Esikleri sil fonksiyonu ile de bulunan değişkenlerdeki değerleri siliyoruz
# Ve new_df olarak atıyoruz.
for col in aykiri:
    new_df = esikleri_sil(df, col)

# Yeni değişkende alt üst eşik haricindeki aykırı değerler yok. Kaç adet kaldığına bakıyoruz
new_df.shape

#Yeni oluşturduğumuz değişken ile eskisinin farkını alıyoruz.
df.shape[0] - new_df.shape[0]


#3) BASKILAMA YONTEMI(re - assignment with thresholds)

df=load_titanic()
df.head()
df.shape

#Alt sınırda düşük , üst sınırdan yüksek değerleri kontrol et
df[((df["Age"] < alt) | (df["Age"] > ust))]["Age"]

# loc yapısı ile ust sınırdan yüksekleri ust sınıra , alt sınırdan düşükleri alt sınıra eşitliyoruz
df.loc[(df["Age"] > ust), "Age"] = ust
df.loc[(df["Age"] < alt), "Age"] = alt

# sonrasında boxplot ile aykırı değerleri kontrol ediyoruz ( yok )
sns.boxplot(df["Age"]);
df=load_titanic()
df.head()


# Aykırı değeri olan değişkenlerin değerlerini alt ve üst sınıra eşitlemek için fonksiyon

def aykiri_baski(veri_seti, degisken):
    ust_deger, alt_deger = aykiri_esikler(veri_seti, degisken)
    veri_seti.loc[(veri_seti[degisken] < alt), degisken] = alt
    veri_seti.loc[(veri_seti[degisken] > ust), degisken] = ust


# fonksiyonu çalıştırıyoruz
aykiri_baski(df, "Age")

# Age değişkenini kontrol ettiğimizde silinmiş.
df[((df["Age"] < alt) | (df["Age"] > ust))]["Age"]

# Fare değişkeni için fonksiyonu çalıştırıyoruz
aykiri_baski(df, "Fare")

# Fare değişkenini kontrol ettiğimizde silinmiş.
df[((df["Fare"] < alt) | (df["Fare"] > ust))]["Fare"]

df = load_titanic()
df.head()

# Aykırı değer olnları kontol ediyoruz ( fakat Fare değişkeni gelmiyor :/  )
aykiri_deger_varmi(df, aykiri)

# Tüm aykırı değerleri aynı anda silmek için fonksiyon yazıyoruz
for col in aykiri:
    aykiri_baski(df, col)


# aykırı değerleri kontrol etmek için çalıştırıyoruz
aykiri_deger_varmi(df, aykiri)


# aykırı gözlem birimi olmadığını görsel ile kontrol edebiliriz.
sns.boxplot(df["Fare"]);

#4. EKSIK DEGER ANALIZI
#Yakala
#Rassallığını
#İncele(Orn kredi kartıharcama, kredi kartı var mı yok mu)
#Problemi Çöz(Sil, Basit Atama, Kırılımlara Göre Atama, Tahmine Dayalı Atama))

df = load_titanic()
df.head()

# Herhangi bir eksik değer olup olmadığını sorguluyoruz
df.isnull().values.any()

# Toplam kaç adet eksik değer
df.isnull().sum().sum()

# Büyükten küçüğe doğru sıralıyoruz
df.isnull().sum().sort_values(ascending=False)

# Oransal olarak eksik değerlerin yüzdeleri
(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)

# En az bir tane olan gözlem birimlerini çekiyoruz
df[df.isnull().any(axis=1)].shape

# Tam olan gözlem birimlerini kontrol ediyoruz
df[df.notnull().all(axis=1)].shape

# Eksik değerleri olanları hızlıca yakalamak için fonksiyon yazıyoruz.
missingvalue = [yok for yok in df.columns if df[yok].isnull().sum() > 0]

# Fonksiyonumuzu çalıştırıyoruz
missingvalue

# Fonksiyon yazarak eksik değerleri bulup, kaç adet olduğunu ve yüzdelerini buluyoruz

def missing_values_table(veri_seti):
    variables_with_na = [col for col in veri_seti.columns if veri_seti[col].isnull().sum() > 0]

    miss = veri_seti[variables_with_na].isnull().sum().sort_values(ascending=False)
    ratio = (veri_seti[variables_with_na].isnull().sum() / veri_seti.shape[0] * 100).sort_values(ascending=False)

    missing_df = pd.concat([miss, np.round(ratio, 2)], axis=1, keys=["miss", "ratio"], names=["Değişken İsimleri"])
    print(missing_df)
    return variables_with_na

# Fonksiyonumuzu çalıştırıyoruz
missing_values_table(df);

# Vahit Hoca'dan Notlar!!
# HIZLI ÇÖZÜM 1: DOKUNMA. NE ZAMAN? AĞAÇ KULLANIYORSAN, AGGREGATION COKSA, TEKILLESTIRME COKSA
# HIZLI COZUM 2: SİLME

# Eksik değer içeren herhangi bir satırı sildiğinde kaç adet satır sileceğini soruyoruz
df.dropna(how="any").shape

# bir satırda tüm değişkenleri eksik veri içerenleri sil diyoruz fakat öyle bir satır yok
df.dropna(how="all").shape

# Eksik değer içeren herhangi bir sütunu sildiğinde kaç adet sütun silineceğini soruyoruz. 12'den 9'a iniyor
df.dropna(axis="columns").shape

# Age değişkenindeki eksik değerlerini Age değişkeninin ortalaması ile dolduruyoruz (median,mode ile de doldurulabilir)
df["Age"].fillna(df["Age"].mean(), inplace=True)

# Age değişkenindeki eksik değerleri kontrol ediyoruz
df["Age"].isnull().sum()

# Bunu anlamadım. Neden mode'un sonuna 0 ekledik. Neden dtype ı kategorik olanı aldık?

df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique() <= 10) else x, axis=0))


#GELISMIS ANALIZLER

df = load_titanic()
df.head()
msno.bar(df);
plt.show()
msno.matrix(df);
msno.heatmap(df);

# Vahit Hoca'dan notlar!!!
# 0.9 bir değişken arttığında diğeride çok şiddetli artar
# -0.9 bir değişken arttığında diğeri azalir.

#EKSIK DEGERLERIN BAGIMLI DEGISKEN ILE ILISKISININ INCELENMESI

# Anlamadım!!!
def missing_vs_target(dataframe, target, variable_with_na):
    temp_df = dataframe.copy()

    for variable in variable_with_na:
        temp_df[variable + "_NA_FLAG"] = np.where(temp_df[variable].isnull(), 1, 0)

        flags_na = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

        for variable in flags_na:
            print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(variable)[target].mean()}), end="\n\n\n")


missing_vs_target(df, "Survived", missingvalue)

#LABEL ENCODING
# - İki sınıflı olan kategorik değişlenlere 1-0 şeklinde label-encoder (binary encoding) uygula. (Cinsiyet)
# - Kategorik değişken nominal ise LABEL ENCODER UYGULANMAZ. (ONE-HOT ENCODING UYGULANIR)
# - Kategorik değişken ordinal ise LABEL ENCODER UYGULANABİLİR. (ONE-HOT ENCODING DE UYGULANABILIR)

df = load_titanic()
df.head()

# Cinsiyet değişkeninde iki farklı gözlem birimi var
print(df["Sex"].head())
len(df["Sex"].value_counts())
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

# Cinsiyet değişkeninin label encoder işlemini gerçekleştiriyoruz. Erkek 1 Kadın 0 olacak şekilde
le.fit_transform(df["Sex"])
le.inverse_transform([0, 1])


# fonksiyon yazarak df içerisindeki 2 değeri olan kategorik değişkenleri encode ediyor
def label_encoder(dataframe):
    labelencoder = preprocessing.LabelEncoder()

    label_cols = [col for col in df.columns if df[col].dtypes == "O"
                  and len(df[col].value_counts()) == 2]
    for col in label_cols:
        df[col] = labelencoder.fit_transform(df[col])
        return df


label_encoder(df)
df["Sex"].head()

#ONE - HOT Encoding

#   - İkiden fazla sınıfa sahip olan kategorik değişkenlerin binary olarak encode edilmesi.

df["Embarked"].value_counts()

# Get Dummies ile değişken içerisindeki değerlere ait yeni sütunlar oluşturuyoruz
pd.get_dummies(df, columns=["Sex"]).head()
pd.get_dummies(df, columns=["Embarked"]).head()

# Drop-first ile ilk değeri siliyoruz. Embarked değeri Q ve S değilse C'dir anlamına geliyor
pd.get_dummies(df, columns=["Embarked"], drop_first=True).head()

# Dummy.na ile eksik veriler için de bir sütun açılıyor
pd.get_dummies(df, columns=["Embarked"], dummy_na=True).head()


# Fonksiyon yazarak 10'dan az değeri olan kategorik değişkenleri one hot encoder ile sütunlar ayırıyoruz.
def one_hot_encoder(dataframe, category_freq=10, nan_as_category=False):
    categorical_cols = [col for col in dataframe.columns if len(dataframe[col].value_counts()) < category_freq
                        and dataframe[col].dtypes == "O"]

    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, dummy_na=nan_as_category, drop_first=True)
    return dataframe.head()

one_hot_encoder(df)

# Fonksiyonumuzu çalıştırmadan önce yapılan tüm işlemlerin silinmesi için veri setimizin ilk halini ekliyoruz
df = load_titanic()
df.head()

# Fonksiyonumuzu çalıştırıyoruz.
# Sex değişkeninde 10'dan az farklı değer olduğu için ayrı sütun açtı.
# Embarked için de ayrı sütun açıldı
# Fakat Sex ve Embarked için ayrı sütun açarken 1.sütunları silindi. Sex için 2 yerine 1, Embarked için 3 yerine 2 sütun açıldı
one_hot_encoder(df)

#Rare Encoding
# Kategorik değişken olan sütunları bulmak için fonksiyon yazıyoruz
cat_cols = [col for col in df.columns if df[col].dtypes == "O"]
cat_cols


# Anlamadım!!!
def cat_summary(data, cat_names):
    for var in cat_names:
        print(var, ":", len(data[var].value_counts()))
        print(pd.DataFrame({"COUNT": data[var].value_counts(),
                            "RATIO": data[var].value_counts() / len(data)}), end="\n\n\n")


cat_summary(df, cat_cols)


# Vahit Hoca'dan Notlar:
# 1. Sınıf Frekansı
# 2. Sınıf Oranı
# 3. Sınıfların target açısından değerlendirilmesi
# 4. rare oranını kendimizin belirleyebilmesi

# ANLAMADIM!!!
def rare_analyser(dataframe, target, rare_perc):
    rare_columns = [col for col in df.columns if df[col].dtypes == "O"
                    and (df[col].value_counts() / len(df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        print(var, ":", len(dataframe[var].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[var].value_counts(),
                            "RATIO": dataframe[var].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(var)[target].mean()}), end="\n\n\n")


rare_analyser(df, "TARGET", 0.001)

# df.groupby("Pclass")["Survived"].mean()
# df.groupby("Pclass").agg({"Survived" : "mean"})

#STANDARTLASTIRMA & DEĞİŞKEN DÖNÜŞÜMLERİ

# StandardScaler: Klasik standartlaştırma. Ortalamayı çıkar, standart sapmaya böl. z = (x - u) / s
df = load_titanic()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(df[["Age"]])
df["Age"] = scaler.transform(df[["Age"]])
df["Age"].describe().T

# count   714.00
# mean      0.00
# std       1.00
# min      -2.02
# 25%      -0.66
# 50%      -0.12
# 75%       0.57
# max       3.47
# Name: Age, dtype: float64



# RobustScaler: Medyanı çıkar iqr'a böl.
df = load_titanic()
from sklearn.preprocessing import RobustScaler

transformer = RobustScaler().fit(df[["Age"]])
df["Age"] = transformer.transform(df[["Age"]])
df["Age"].describe().T

# count   714.00
# mean      0.10
# std       0.81
# min      -1.54
# 25%      -0.44
# 50%       0.00
# 75%       0.56
# max       2.91
# Name: Age, dtype: float64





# MinMaxScaler: Verilen 2 değer arasında değişken dönüşümü
# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std * (max - min) + min
df = load_titanic()
from sklearn.preprocessing import MinMaxScaler

transformer = MinMaxScaler((-10, 10)).fit(df[["Age"]])
df["Age"] = transformer.transform(df[["Age"]])  # on tanımlı değeri 0 ile 1 arası.
df["Age"].describe().T

# count   714.00
# mean     -2.64
# std       3.65
# min     -10.00
# 25%      -5.05
# 50%      -3.07
# 75%      -0.56
# max      10.00
# Name: Age, dtype: float64



# Log: Logaritmik dönüşüm.
# not: - degerler varsa logaritma alınamayacağı için bu durum göz önünde bulundurulmalı.
df = load_titanic()
df["Age"] = np.log(df["Age"])
df["Age"].describe().T

#FEATURE ENGINEER
# FLAG, BOOL
df = load_titanic()
df.head()

df["NEW_CABIN_BOOL"] = df["Cabin"].notnull().astype('int')
# df["NEW_CABIN_BOOL2"] = df["Cabin"].isnull().astype('int')


df.loc[((df['SibSp'] + df['Parch']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SibSp'] + df['Parch']) == 0), "NEW_IS_ALONE"] = "YES"
df.head()

# LETTER COUNT
df["NEW_NAME_COUNT"] = df["Name"].str.len()

# WORD COUNT
df["NEW_NAME_WORD_COUNT"] = df["Name"].apply(lambda x: len(str(x).split(" ")))
df["NEW_NAME_DR"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
df["NEW_NAME_DR"].head()
df["NEW_NAME_DR"].mean()
df.groupby("NEW_NAME_DR").agg({"Survived": "mean"})
df["NEW_NAME_DR"].value_counts()



df["Age"].mean()
df['NEW_TITLE'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
df.head()
df.groupby("NEW_TITLE").agg({"Age": "mean"})
df[["NEW_TITLE", "Survived", "Age"]].groupby(["NEW_TITLE"]).agg({"Survived": "mean", "Age": ["count", "mean"]})

# NUMERIC TO CATEGORICAL
df.loc[(df['Age'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['Age'] >= 18) & (df['Age'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['Age'] >= 56), 'NEW_AGE_CAT'] = 'senior'

# INTERACTIONS

df["NEW_FAMILY_SIZE"] = df["SibSp"] + df["Parch"] + 1
df["NEW_AGExPCLASS"] = df["Age"] * df["Pclass"]
df.loc[(df['Sex'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['Sex'] == 'male') & ((df['Age'] > 21) & (df['Age']) < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['Sex'] == 'male') & (df['Age'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['Sex'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['Sex'] == 'female') & ((df['Age'] > 21) & (df['Age']) < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'
df["Age"] = df["Age"].fillna(df.groupby("NEW_TITLE")["Age"].transform("median"))
