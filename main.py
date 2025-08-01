#Nesne Tespiti
#------------------------------------------------------------------------------------------------------------------------
# from ultralytics import YOLO
# import cv2

# # Eğitilmiş modelini yükle (senin model yolunu gir!)
# model = YOLO(r"C:\Users\eness\Desktop\görsel_boyut_kucultme\runs\detect\train4\weights\best.pt")


# # Test edilecek görselin yolunu belirt
# image_path = r"DSCF0174.JPG"

# # Görseli oku
# image = cv2.imread(image_path)

# # Modelle tahmin yap
# results = model.predict(image, conf=0.25,save = True)  # İstersen conf değerini düşürebilirsin

# # Tahmin sonuçları
# for result in results:
#     boxes = result.boxes
#     for box in boxes:
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         conf = float(box.conf[0])
#         cls_id = int(box.cls[0])
#         label = model.names[cls_id]

#         # Dikdörtgen çiz
#         cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
        
#         # Etiket metni
#         text = f"{label} {conf:.2f}"
#         cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
#                     0.5, (255, 255, 255), 2)

# # Sonucu göster
# cv2.imshow("Tahmin Sonucu", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Kaydetmek istersen:
# cv2.imwrite("output.jpg", image)




#Nesneyi tespit eder nesnenin alttan %13(?)lük kısmında yeşil piksel arar
#---------------------------------------------------------------------------------------------------------------------------------
# import numpy as np
# import cv2
# from ultralytics import YOLO

# # Modeli yükle
# model = YOLO(r"runs\detect\train8\weights\best.pt")

# # Görseli yükle
# image_path = "DSCF0372.JPG"
# image = cv2.imread(image_path)

# # Tahmin yap
# results = model.predict(image, conf=0.1,save= True)

# # Her tespit için işlem

# for result in results:
#     boxes = result.boxes
#     for box in boxes:
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         ALT_PERCENT = 0.13  # Alt kısmın yüzdesi
#         box_height = y2 - y1
#         alt_y1 = y2 - int(box_height * ALT_PERCENT)
#         alt_y2 = y2

#         # Sadece alt kısmı al
#         alt_bolum = image[alt_y1:alt_y2, x1:x2]

#         # Yeşil alanı bul ve kırmızıya çevir (RGB değil, OpenCV'de BGR)
#         lower_green = np.array([0, 87, 0])
#         upper_green = np.array([110, 255, 110])
#         mask = cv2.inRange(alt_bolum, lower_green, upper_green)
#         alt_bolum[mask > 0] = [0, 0, 255]

#         # Değiştirilen alt kısmı görsele geri yerleştir
#         image[alt_y1:alt_y2, x1:x2] = alt_bolum

#         # Alt kısmı kutu içine al (isteğe bağlı)
#         cv2.rectangle(image, (x1, alt_y1), (x2, alt_y2), (0, 0, 255), 2)
# resized_image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
# #cv2.namedWindow("window", cv2.WINDOW_NORMAL)
# cv2.imshow("Sonuc", resized_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()




# import numpy as np
# import cv2
# from ultralytics import YOLO

# # Modeli yükle
# model = YOLO(r"runs/detect/train6/weights/best.pt")

# # Görseli yükle
# image_path = "agacli_direk.png"
# image = cv2.imread(image_path)

# # Tahmin yap
# results = model.predict(image, conf=0.2, save=False)

# # Sabitler
# ZEMIN_MARGIN_PX = 75  # Direğin altına bakılacak piksel yüksekliği
# MIDBOX_RATIO = 0.25   # Kutu genişliğinin yanlardan kırpılacak yüzdesi
# LOWER_GREEN = np.array([0, 85, 0])
# UPPER_GREEN = np.array([95, 255, 95])
# HIGHLIGHT_COLOR = [0, 0, 255]  # Kırmızı (BGR)

# # Tespit edilen her nesne için işlem
# for result in results:
#     for box in result.boxes:
#         x1, y1, x2, y2 = map(int, box.xyxy[0])

#         # Yükseklik kontrolü: Görsel dışına çıkma
#         alt_y2 = y2
#         alt_y1 = max(y2 - ZEMIN_MARGIN_PX, 0)

#         # Orta %70'lik kısmı al (yanlardan %15 kırp)
#         margin_x = int((x2 - x1) * MIDBOX_RATIO)
#         x_mid1 = x1 + margin_x
#         x_mid2 = x2 - margin_x

#         # Alt kısmın merkezini al
#         alt_merkez_bolum = image[alt_y1:alt_y2, x_mid1:x_mid2]

#         # Yeşil pikselleri maskele ve kırmızıya boya
#         mask = cv2.inRange(alt_merkez_bolum, LOWER_GREEN, UPPER_GREEN)
#         alt_merkez_bolum[mask > 0] = HIGHLIGHT_COLOR

#         # Güncellenmiş kısmı görsele yerleştir
#         image[alt_y1:alt_y2, x_mid1:x_mid2] = alt_merkez_bolum

#         # Görselleştirme: alt bölgeye kutu
#         cv2.rectangle(image, (x_mid1, alt_y1), (x_mid2, alt_y2), HIGHLIGHT_COLOR, 1)

# # Görseli yeniden boyutlandır ve göster
# resized_image = cv2.resize(image, (0, 0), fx=0.8, fy=0.8)
# cv2.imshow("Sonuc", resized_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()





#Nesneyi algılayıp yuvarlak içine almak için
#----------------------------------------------------------------------------------------------------------------------------
# import cv2

# # Görsel ve .txt yolu
# image_path = "voltage_line.v5i.yolov8/train/images/DSCF0284.JPG"
# txt_path = "voltage_line.v5i.yolov8/train/labels/DSCF0284.txt"

# # Görseli oku
# image = cv2.imread(image_path)
# height, width, _ = image.shape

# # .txt dosyasını oku
# with open(txt_path, "r") as f:
#     lines = f.readlines()

# # Her satırı işle ve kutu çiz
# for line in lines:
#     parts = line.strip().split()
#     class_id, x_center, y_center, w, h , conf = map(float, parts)
    
#     # Normalize değerleri piksele çevir
#     x_center *= width
#     y_center *= height
#     w *= width
#     h *= height

#     # Kutu köşelerini hesapla
#     x1 = int(x_center - w / 2)
#     y1 = int(y_center - h / 2)
#     x2 = int(x_center + w / 2)
#     y2 = int(y_center + h / 2)

#     # Kutuyu çiz
#     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     cv2.putText(image, f"{int(class_id)}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
#                 0.6, (0, 255, 0), 2)

# # Sonucu göster
# cv2.imshow("Etiketli Görsel", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()






import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO(r"runs\detect\train10\weights\best.pt")  # Model yolunu gir

image_path = "agacli_direk.png"
image = cv2.imread(image_path)

results = model.predict(image, conf=0.1,save=True)

ALT_PERCENT = 0.15  # Alt kısmın yüzdesi

for result in results:
    for box in result.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        if label == "direk":
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            box_height = y2 - y1
            alt_y1 = y2 - int(box_height * ALT_PERCENT)
            alt_y2 = y2
            alt_bolum = image[alt_y1:alt_y2, x1:x2]

            # Alt kısımda tekrar tespit yap
            alt_results = model.predict(alt_bolum, conf=0.25)
            for alt_result in alt_results:
                for alt_box in alt_result.boxes:
                    alt_cls_id = int(alt_box.cls[0])
                    alt_label = model.names[alt_cls_id]
                    if alt_label == "tree":
                        # Alt bölgedeki ağacı orijinal görselde kutu içine al
                        ax1, ay1, ax2, ay2 = map(int, alt_box.xyxy[0])
                        cv2.rectangle(image, (x1 + ax1, alt_y1 + ay1), (x1 + ax2, alt_y1 + ay2), (0, 255, 0), 2)

cv2.imshow("Sonuc", image)
cv2.waitKey(0)
cv2.destroyAllWindows()