# Computer Vision Mini Project 2 — Scene Classification

มินิโปรเจกต์จำแนกภาพฉากด้วย EfficientNetB0 ใช้ transfer learning และทำหน้าเว็บด้วย Gradio สำหรับอัปโหลดภาพดูผลทำนาย

จำแนกภาพเป็น 6 กลุ่ม: buildings, forest, glacier, mountain, sea และ street

## ลองใช้โมเดล

มีไฟล์โมเดลที่ฝึกแล้ว `scene_model.keras` และชื่อคลาส `class_indices.json` อยู่ใน repo ใช้ Python 3.10:

```sh
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

เปิดลิงก์ localhost ที่แสดงใน terminal แล้วอัปโหลดภาพ ระบบจะแสดงผลทำนาย 3 อันดับแรก

## การฝึกโมเดล

ใช้ภาพขนาด 224 × 224, batch size 16 และ EfficientNetB0 ที่เริ่มจากน้ำหนัก ImageNet
ตรึงชั้นของโมเดลฐาน แล้วเพิ่ม Global Average Pooling, Dense 128 และ Dropout 0.3 ก่อนชั้นจำแนก 6 คลาส

![กราฟจากประวัติการฝึก](training-history.png)

กราฟมาจาก train_history.json ของการฝึกที่บันทึกไว้ ค่า validation เป็นผลระหว่างฝึก ไม่ใช่ผลทดสอบชุดใหม่

| ไฟล์ | หน้าที่ |
|---|---|
| app.py | หน้าอัปโหลดภาพและทำนาย |
| train.py | ฝึกและบันทึกโมเดล |
| evaluate.py | ประเมินชุด test และแสดง confusion matrix |
| make_val.py | แบ่งภาพจาก train ไป val โดยย้ายไฟล์ |

หากจะฝึกเอง เตรียม `dataset/train`, `dataset/val` และ `dataset/test` โดยแต่ละโฟลเดอร์มีโฟลเดอร์ย่อยตามชื่อคลาส แล้วรัน `python train.py` หรือ `python evaluate.py`
ชุดภาพฝึกไม่ได้รวมใน repo; การลองทำนายด้วยโมเดลที่ให้มาไม่ต้องใช้ dataset
ใช้ make_val.py กับสำเนาชุดข้อมูลเท่านั้น เพราะสคริปต์ย้ายไฟล์จาก train

[งานอื่นในวิชา Computer Vision](https://github.com/Painter121/computer-vision-homework)
