## Assignment 1
说明：由于部分模型文件大小超过100MB，超过Github单文件上限，无法上传。

### Project Structure
```
Ex1
 ├─configs : include all the configuration file
 │      dense161_custom.py 
 │      inception_v3_custom.py
 │      mobilenet_v3_lager_custom.py
 │      resnet50_custom.py
 │      resnet50_custom_0.py : baseline
 │      resnet50_custom_1.py : frozen_stages=1
 │      resnet50_custom_1e-2.py : lr = 1e-2
 │      resnet50_custom_1e-3.py : lr = 1e-2
 │      resnet50_custom_1e-4.py : lr = 1e-2
 │      resnet50_custom_2.py : frozen_stages=2
 │      resnet50_custom_3.py : frozen_stages=3
 │      resnet50_custom_4.py : frozen_stages=4
 │      resnext50_custom.py
 │
 └─work_dirs : include the lastest saved trained model
```
```
Ex2
├── assignment1.pdf : our report
├── best_model.pth : the saved trained model
└── main.py : our completed script file
```