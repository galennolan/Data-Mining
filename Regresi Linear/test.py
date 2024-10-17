from manim import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

class SpamDetectionAnimation(Scene):
    def construct(self):
        # Langkah 1: Membuat Tabel Preview Data
        # Memuat dataset
        df = pd.read_csv('../csv/SMSSpamCollection.csv', sep='\t', names=['label', 'message'])

        # Menampilkan beberapa data dalam bentuk tabel
        table_data = df.head().values.tolist()
        col_labels = [Text("Label"), Text("Message")]
        table = Table(
            table_data,
            col_labels=col_labels,
            h_buff=1,
            v_buff=0.5,
            include_outer_lines=True
        ).scale(0.4)

        # Mengatur opacity garis tabel dan menaikkan posisi tabel
        for line in table.get_vertical_lines() + table.get_horizontal_lines():
            line.set_opacity(0.5)

        table.move_to(ORIGIN).shift(UP * 2)
        self.play(Create(table))
        self.wait(2)
        self.play(FadeOut(table))

        # Langkah 2: Membuat Grafik Kartesian untuk Visualisasi Data
        # Mengganti label 'ham' dengan 0 dan 'spam' dengan 1
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})

        # Memisahkan fitur dan label
        X = df['message']
        y = df['label']

        # Membagi data menjadi data latih dan data uji
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Mengubah teks menjadi fitur numerik menggunakan TF-IDF Vectorizer
        vectorizer = TfidfVectorizer()
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        # Membuat grafik kartesian
        axes = Axes(
            x_range=[0, 10, 1],  # Rentang sumbu X
            y_range=[0, 10, 1],  # Rentang sumbu Y
            axis_config={"include_numbers": True},
        ).scale(0.7)
        
        x_label = axes.get_x_axis_label(Tex("X"), edge=RIGHT, direction=DOWN, buff=0.4)
        y_label = axes.get_y_axis_label(Tex("Y"), edge=UP, direction=LEFT, buff=0.4)
        
        axes.move_to(ORIGIN).shift(DOWN * 1.5)
        self.play(Create(axes), Write(x_label), Write(y_label))
        self.wait(1)
        
        # Membuat dan melatih model Logistic Regression
        model = LogisticRegression()
        model.fit(X_train_tfidf, y_train)

        # Melakukan prediksi pada data uji
        y_pred = model.predict(X_test_tfidf)

        # Menghitung akurasi model
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_text = Text(f"Akurasi: {accuracy * 100:.2f}%", color=YELLOW).to_edge(UP).shift(DOWN * 0.5)
        
        # Menampilkan persamaan regresi dan akurasi
        self.play(Write(accuracy_text))
        self.wait(2)
        self.play(FadeOut(accuracy_text))

        # Menampilkan laporan klasifikasi
        report = classification_report(y_test, y_pred, output_dict=True)
        report_table_data = [
            [key] + list(map(lambda x: f"{x:.2f}", value.values()))
            for key, value in report.items() if isinstance(value, dict)
        ]
        report_table = Table(
            report_table_data,
            col_labels=[Text("Class"), Text("Precision"), Text("Recall"), Text("F1-score"), Text("Support")],
            h_buff=1,
            v_buff=0.5,
            include_outer_lines=True
        ).scale(0.4)
        report_table.move_to(ORIGIN)
        self.play(Create(report_table))
        self.wait(2)

# Konfigurasi resolusi video (1080x1920)
config.pixel_height = 1920
config.pixel_width = 1080
config.frame_height = 14.4
config.frame_width = 8.1

if __name__ == "__main__":
    scene = SpamDetectionAnimation()
    scene.render()
