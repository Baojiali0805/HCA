
from ultralytics import YOLO

if __name__ == '__main__':
    # model.load('yolo11n.pt') # 加载预训练权重,改进或者做对比实验时候不建议打开，因为用预训练模型整体精度没有很明显的提升
    model = YOLO('yolo11-HCA.yaml')
    model.train(data='D:/learning/1.1/datasets-train/datasetsv3/FallDown.yaml',
                imgsz=640,
                epochs=300,
                batch=8,
                workers=0,
                device=0,
                optimizer='SGD',
                close_mosaic=10,
                resume=False,
                project='runs/train',
                name='exp',
                single_cls=False,
                cache=False,
                )



