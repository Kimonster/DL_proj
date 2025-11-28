# 工作要求
## 1. 数据预处理
目前项目所需的数据在processe_set文件夹中，数据的结构你可以自己阅读文件查看，提示：
- cover drop move都是不同的动作
- 每个动作文件夹中有对应的index.json文件，里面有每个同级文件夹的信息

我需要你完成以下工作：
- 我的每个视频文件夹中有两个文件夹，一个是input，一个是target，input中有20张图片，target中有1张图片。我想利用前20帧图片作为输入，最后一张图片作为目标输出，利用instructpix2pix模型进行预测。
- 所以我需要你先利用“残影法”将前20帧图片聚合成一张图片，具体示范代码如下

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_synthetic_ghosting_example():
    # 1. 创建三张空白画布 (128x128)
    frames = []
    # 模拟三个时间点：t-10, t-5, t (当前帧)
    positions = [(30, 30), (60, 60), (90, 90)] # 小球位置：左上 -> 右下

    for i, pos in enumerate(positions):
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        # 画一个白色背景
        img.fill(240)
        # 画一个红色小球
        cv2.circle(img, pos, 15, (0, 0, 255), -1)
        # 加一点字模拟真实场景
        cv2.putText(img, f"Frame {i}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        frames.append(img)

    # 2. 执行加权融合 (残影法)
    # 权重：当前帧(t)占 60%，上一帧(t-5)占 30%，上上帧(t-10)占 10%
    w1, w2, w3 = 0.1, 0.3, 0.6

    # 先融合前两帧
    blended_past = cv2.addWeighted(frames[0], w1 / (w1 + w2), frames[1], w2 / (w1 + w2), 0)
    # 再融合当前帧
    final_input = cv2.addWeighted(blended_past, 1 - w3, frames[2], w3, 0)

    # 3. 展示结果
    plt.figure(figsize=(10, 4))

    titles = ["Past (t-10)", "Past (t-5)", "Current (t)", "INPUT for IP2P"]
    images = frames + [final_input]

    for i in range(4):
        plt.subplot(1, 4, i+1)
        # OpenCV 是 BGR，Matplotlib 是 RGB，转一下色
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.savefig("ghosting_demo.png")
    print("样例已保存为 ghosting_demo.png")
    plt.show()

if __name__ == "__main__":
    create_synthetic_ghosting_example()
```
- 然后将聚合之后的图片作为真正的input，target文件夹中的图片作为目标输出，index.json文件中的label值作为prompt，利用instructpix2pix模型进行预测。

## 2. 模型微调
我的原始的模型权重之后将会存储在models文件夹中，你设置路径去这里读取模型权重即可

需要做的工作

- 我使用的是train_instruct_pix2pix.py脚本进行模型微调，你需要利用上一步生成的input图片，和target图片与index.json中的label值进行微调，你可以自己建立新的文件夹存储数据，并且设置正确的数据读取路径。

- train_instruct_pix2pix.py中进行的模型的全量微调，我想使用lora微调，你需要合理修改train_instruct_pix2pix.py脚本，使用lora进行微调。

- 在训练的时候定期存储模型权重。

- 在训练的时候定期存储training loss和validation loss，同时，还要计算模型的PSNR和SSIM指标，并将这些指标存储到一个csv文件中，最后进行可视化。

- 训练完成后，生成一个README.md文件，记录模型的训练过程、参数设置、最终的模型性能指标等信息。

## 3. 模型推理

在模型微调完成后，编写一个推理脚本，利用微调后的模型进行推理，输入是新的input图片，输出是预测的target图片。