# LipReading
2019年“创青春·交子杯”新网银行高校金融科技挑战赛-AI算法赛道唇语识别思路和代码分享

比赛网址:<https://www.dcjingsai.com/common/cmpt/2019%E5%B9%B4%E2%80%9C%E5%88%9B%E9%9D%92%E6%98%A5%C2%B7%E4%BA%A4%E5%AD%90%E6%9D%AF%E2%80%9D%E6%96%B0%E7%BD%91%E9%93%B6%E8%A1%8C%E9%AB%98%E6%A0%A1%E9%87%91%E8%9E%8D%E7%A7%91%E6%8A%80%E6%8C%91%E6%88%98%E8%B5%9B-AI%E7%AE%97%E6%B3%95%E8%B5%9B%E9%81%93_%E7%AB%9E%E8%B5%9B%E4%BF%A1%E6%81%AF.html>  
主要基于论文“Combining Residual Networks with LSTMs for Lipreading”实现
# 成绩
初赛提交的原始代码单折0.65，十折之后B榜线上成绩0.79，位列第十.  
项目中的初赛代码是在初赛完成之后又进行了数据处理方面的优化的，验证集单折0.67，预计十折后线上成绩>0.8  
决赛验证集单折约0.69，八折后线上成绩0.87，位列第六。  
 ***如有帮助还请点个star***

# 环境需求
torch==1.2.0  
face-alignment==1.0.0  

# 一、初赛
&ensp;&ensp;&ensp;&ensp;一组图像序列对应一个词语，由于词语之间没有必然联系，所以可以看作一个纯分类问题。难点在于数据处理，样本不同，图片的数量和大小都不同，并且数据明显做了抽帧处理，试图根据图片数量来确定词语长度应该是没什么效果的。  
## 数据处理
这里参考了FesianXu的代码（https://github.com/FesianXu/LipNet_ChineseWordsClassification），使用face-alignment库做嘴唇区域切割。  
![avatar](https://github.com/liuzhejun/XWbank_LipReading/blob/master/README_IMGS/2d3d.png)  
&ensp;&ensp;&ensp;&ensp;使用face-alignment做嘴唇切割比较耗时，很依赖于cup处理能力，我处理完所有样本大概在3小时左右，并且许多图片由于人脸不完整，无法正确识别，但由于帧与帧之间有相关性，可以通过上下帧的嘴部位置确定当前帧的嘴部位置。  
&ensp;&ensp;&ensp;&ensp;另一种方法是人工标注一部分嘴部区域图片，再训练一个专门识别嘴部区域的模型，以识别出其余图片的嘴唇区域，个人更推荐这种方法，只是由于时间问题没有尝试。
## 模型
&ensp;&ensp;&ensp;&ensp;在答辩的时候发现很多选手用的模型都比较类似，`3D卷积`+`ResNet`+[`RNN`|`TSM`]就能达到比较好的效果：
![avatar](https://github.com/liuzhejun/XWbank_LipReading/blob/master/README_IMGS/model_1.png)  
在我的代码中相对于论文作者的模型参数，我做了如下修改：  
* 图像输入大小从1x112x112改为3x120X180，也就是将灰度图改为了彩色图，并将图像大小调整为了更适合嘴唇大小的120x180（但1x112x112也能得到0.65的单折验证集成绩）.
* 3D卷积核大小由5x7x7改为3x5x5，一方面是为了减少模型参数，另一方面是我们的每个样本帧数是比较少的，跨越5个time step的空间卷积有些大了。
* 减少ResNet层数：一方面为了减少模型参数、使模型更容易下降，另一方面由于数据集较小，过大的模型容易过拟合。
* 将2层双向LSTM改为单层双向GRU：同样为了减少模型参数，防止过拟合。  

&ensp;&ensp;&ensp;&ensp;此外还有一个小trick，也就是图片中提到的线性分类层中的自适应词语边界，实际上就是不将GRU最后输出的隐藏层向量直接连接分类层，而是将GRU的每个time step的输出连接全连接层，在做sotfmax之后在time step维度相加，使得每个time step的输出都能为最后的分类做出贡献，原因就在于词语的边界位未知，不一定最后一帧图片刚好表示词语说完。

## 其他
&ensp;&ensp;&ensp;&ensp;模型的调参非常重要，提分的过程无非就是在和过拟合做斗争的过程，所以我在模型额外添加了dropout，在训练时添加了正则。  
&ensp;&ensp;&ensp;&ensp;此外，我在数据处理时先将样本按帧数排序，以保证每个batch中的数据填充最少，但为了防止模型对样本帧数产生依赖，又要乱序feed进模型，这前后的准确率相差大概30%。  

## 代码使用
### 1.获取landmarks
使用face-alignment库提取训练集的面部坐标数据
```shell
python get_landmarks --root_dir 新网银行唇语识别竞赛数据/1.训练集/lip_train/
                     --save_path data/train_landmarks.dat
```
提取测试集的面部坐标数据
```shell
python get_landmarks --root_dir 新网银行唇语识别竞赛数据/2.测试集/lip_test/
                     --save_path data/test_landmarks.dat
```
程序会读取`root_dir`下的每个文件夹中的每张图片，提取面部特征点数据，保存为`save_path`  

### 2.数据处理
```shell
python data_process_with_face-alignment --train_path 新网银行唇语识别竞赛数据/1.训练集/lip_train/
                                        --test_path 新网银行唇语识别竞赛数据/2.测试集/lip_test/
                                        --label_path 新网银行唇语识别竞赛数据/1.训练集/lip_train.txt
                                        --train_landmarks_path data/train_landmarks.dat
                                        --test_landmarks_path data/test_landmarks.dat
                                        --save_path data/
                                        --k 5
```
程序读取训练集和测试集图片，根据上一步保存的`train_landmarks.dat`和`test_landmarks.dat`面部特征点数据进行嘴部区域切割，并进行归一化、按帧数排序、处理词表、产生标签，最后保存k个`.dat`文件和一个`测试集数据`文件以及`vocab.txt`到`save_path`目录下。  

### 3.训练
```shell
python train.py --data_path data/train_data.dat
                --test_data_path data/test_data.dat
                --vocab_path data/vocab.txt
                --model_save_path model/
                --batch_size 32
                --epochs 40
                --k 5
```
程序会读取上一步处理的数据集和训练集文件，并根据`batch_size`填充数据，输入模型进行训练，总共训练`k`次。每次训练完成会保存一个模型权重文件`fold_k_model.pt`和一个测试集的预测结果`fold_k_result.pkl`到`model_save_path`目录下，完成所有训练后自动进行预测，并将预测结果保存为`submit.txt`  

# 二、决赛
决赛的改变主要在于数据，唇语数据由互不相关的词语变为`000-999`之间的1000个数字，且规定说出每个数的方式为`一二三`，而非`一百二十三`，训练集样本数量由9996变为约5000个，难点就在于分类数的增加、样本数量的减少。  
## 思路
&ensp;&ensp;&ensp;&ensp;如果直接跑1000分类，那么线下准确率只有0.35左右，必然不行。一种容易想到的思路就是将1000分类转化为3个10分类，其实这种方法是可行的，有队伍训练3个模型分别识别个、十、百位数三个数字，最终准确率达到了0.9+，的确让我比较惊讶，而我虽然也尝试过3个10分类，但我是将3个分类压缩在一个模型。也就是3个10分类的权重是共享的，也许是参数数量限制了我的准确率，这种做法最后准确率只有0.25.  
&ensp;&ensp;&ensp;&ensp;最后我的思路是这样，尽管可以将问题分解为3个10分类问题，但最终还是要将3个分类概率相乘，其实仍旧是1000分类，只是相对于1000分类而言，5000个样本的训练集是能够满足10分类的要求的，本质上就是要让模型意识到不同样本之间并不是毫无联系的两类，而是所有的样本都是可以划分为个、十、百三个位置的10分类，且个、十、百三个位置之间也是相同的10分类问题。  
&ensp;&ensp;&ensp;&ensp;所以我在模型仍是1000分类的基础上，在Resnet层之后添加了一个`Attention`分支，用来做3个10分类，最终的loss是1000分类和10分类的两个loss之和。目的就是为了让前面的3D卷积和ResNet层学习到单个数字的唇语特征，而非毫无相关的1000分类。  
![avatar](https://github.com/liuzhejun/XWbank_LipReading/blob/master/README_IMGS/model_2.png)    
&ensp;&ensp;&ensp;&ensp;采用`注意力机制`是因为一个样本中的每帧图片分别对个、十、百位数字的分类贡献必然不同，分别将`ResNet`中的每个time stpe的隐藏层向量以不同权重相加，即代表每个数字对不同帧的图像的注意程度。  
&ensp;&ensp;&ensp;&ensp;模型准确率应该还有提升的空间，因为相对于第一个分支的GRU层，`Attention`层的参数是非常少的，将`单头注意力`改为`多头注意力`应该会有提升。

## 代码
&ensp;&ensp;&ensp;&ensp;代码的使用方式和初赛代码没有任何区别，因为仅仅改变了模型的loss计算方式，对模型外部而言，唯一的改变就是训练集`label`有微小改变，初赛中训练集的`label`是由样本的id（形如‘`00b60f1b01138fbf902bd4bee2d7ebc1`’）得到词语（形如‘`快乐`’），再由词语得到类别下标。而决赛中由样本id得到的词语(形如‘`123`’)本就是一个数字，可以代表类别下标

# 三、总结
&ensp;&ensp;&ensp;&ensp;本人是NLP方向，非专攻CV领域，有勘误之处还请指出，很多优化和改进都是个人理解和尝试出来的，有可能理论上有所偏颇，总的来说这次比赛收获颇丰，学习到和很多新东西，也意识到自己和别人的差距，希望能够更加进步吧。

