# from models.yolo import Model

# if __name__ == '__main__':
#     # choose your yaml file
#     model = Model(r'/root/kxl/yolov5-master/models/yolov5s-MYDowndw.yaml')
#     model.info(verbose=False, img_size=[640, 640])
#     model.fuse()
import torch  
import matplotlib.pyplot as plt  
from torchinfo import summary  # pip install torchinfo  

def visualize_model_params(pt_file):  
    # 加载模型  
    model = torch.load(pt_file, map_location=torch.device('cpu'))  # 加载到CPU  
    if 'model' in model:  # 检查是否是YOLO格式  
        model = model['model'].float()  # 提取模型  

    # 获取模型参数统计信息  
    model_summary = summary(model, input_size=(1, 3, 640, 640), verbose=0)  # 假设输入大小为 640x640  
    layer_names = []  
    param_counts = []  

    # 遍历每一层，提取名称和参数量  
    for layer in model_summary.summary_list:  
        layer_names.append(layer.class_name)  
        param_counts.append(layer.num_params)  

    # 绘制柱状图  
    plt.figure(figsize=(12, 6))  
    plt.bar(range(len(param_counts)), param_counts, color='skyblue')  
    plt.xticks(range(len(param_counts)), layer_names, rotation=90, fontsize=8)  
    plt.xlabel('Layer Name')  
    plt.ylabel('Number of Parameters')  
    plt.title('Parameter Count per Layer in YOLO Model')  
    plt.tight_layout()  
    # 保存柱状图  
    plt.savefig("bar_chart_params.png")  # 保存为当前目录下的 bar_chart_params.png  
    plt.show()  

    # 绘制饼图（可选）  
    plt.figure(figsize=(8, 8))  
    plt.pie(param_counts, labels=layer_names, autopct='%1.1f%%', startangle=140)  
    plt.title('Parameter Distribution in YOLO Model')  
    plt.tight_layout()  
    # 保存饼图  
    plt.savefig("pie_chart_params.png")  # 保存为当前目录下的 pie_chart_params.png  
    plt.show()  

# 使用方法  
pt_file = "/root/kxl/yolov5-master/runs/train/exp-v5/weights/best.pt"  # 替换为你的.pt文件路径  
visualize_model_params(pt_file)