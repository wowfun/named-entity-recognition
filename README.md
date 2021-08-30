# named-entity-recognition

## 运行环境
tqdm  
python==3.8  
pytorch>=1.7.1  
tensorboard==2.5.0  
tensorboardx==2.2

## 运行步骤
### BiLSTM-CRF
运行 `main_bilstm_crf.ipynb` 即可。
### BiLSTM
运行 `main_bilstm.ipynb` 即可。

## 文件结构
- checkpoints/  # 模型保存目录
- data/
    - ResumeNER/ # 简历实体数据集
- logs/ # 运行日志保存目录（logger输出、tensorboard输出）
- models/ # 模型目录
- utils/ # 辅助函数目录
- solver.py # 模型训练和评估封装
- main_bilstm_crf.ipynb / main_bilstm.ipynb # 主函数运行

## Ref
- [named_entity_recognition](https://github.com/sqkika/named_entity_recognition)