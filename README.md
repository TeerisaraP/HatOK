
# HatOK ⛑️👌🏻

Safety Helmet Detection System for High-Risk Areas Using AI Technology\
ระบบตรวจจับการใส่หมวกเซฟตี้เพื่อเพิ่มความปลอดภัยในพื้นที่เสี่ยง ด้วยเทคโนโลยี AI
## Concepts
ระบบที่ใช้ Deep Learning และ Computer Vision แยกแยะและตรวจจับการใส่หมวกเซฟตี้ในพื้นที่ต่างๆ เช่น พื้นที่ก่อสร้าง พื้นที่อันตราย หรือพื้นที่ที่จำเป็นต้องใช้หมวกเซฟตี้


## Related Theories
ในงานวิจัยนี้ ได้นำเอาหลักการและทฤษฎีจากหลายสาขามาประกอบกัน เพื่อพัฒนาระบบตรวจจับการสวมหมวกเซฟตี้ให้มีความแม่นยำและตอบสนองได้อย่างรวดเร็ว ซึ่งแนวคิดเหล่านี้ประกอบด้วยการประยุกต์ใช้เทคโนโลยีปัญญาประดิษฐ์ (AI), คอมพิวเตอร์วิทัศน์ (Computer Vision), การเรียนรู้เชิงลึก (Deep Learning) การตรวจจับวัตถุ (Object Detection), การประมวลผลภาพ (Image Processing) 

## Dataset
[Roboflow HatOK-helmet detection](https://universe.roboflow.com/hatok/hatok-helmet-detection)\
dataset มีจำนวน 1968 รูป และเข้าสู่กระบวนการ Annotate เพื่อแบ่งคลาสให้กับ dataset \
ทั้งหมด 5 คลาส ได้แก่
- white helmet จำนวน 840 รูป
- yellow helmet จำนวน 754 รูป
- blue helmet จำนวน 652 รูป
- red helmet จำนวน 356 รูป
- head จำนวน 383 รูป

## Model
SSD-MobileNet, YOLO

## Work process

1. รับภาพจากกล้อง CCTV หรือ ไฟล์ภาพ
2. ใช้โมเดลสำหรับการตรวจจับ ในการตรวจจับการใส่หมวกเซฟตี้
3. แสดงผล Bounding Box และจำแนกสีของหมวก กับผู้ไม่ใส่หมวก
4. บันทึกข้อมูลที่ตรวจจับ


## Authors

- [@TeerisaraP](https://github.com/TeerisaraP)
- [@akitmooyu26](https://github.com/akitmooyu26)
