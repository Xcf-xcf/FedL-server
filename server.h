#ifndef SERVER
#define SERVER
#include<iostream>
#include<string>
#include<vector>
#include<exception>
#include<algorithm>
#include "model.h"
#include <winsock2.h>
#include <iomanip>
#pragma comment(lib, "ws2_32.lib")
#pragma warning(disable:4996)
using namespace std;
#define void LDA
#define MaxClientNum 100
#define BufSize 10000
//主线程模拟服务端
class server
{
private:
	string         id;                             //主线程id
	SOCKET         s;
	SOCKET         c;
	vector<string> clt_id;                         //存储客户端线程标识
	int            global_epochs;                  //参数聚合次数 即参数上传/下发次数
	int            local_epochs;                   //客户端模型训练次数（针对神经网络）
	vector<int>    batch_size;                     //客户端每次训练所使用的数据集大小（实际中可能每个客户端每次训练集大小不相同,参数聚合时需要该参数进行加权）
	int            k;                              //每轮挑选k个客户端线程进行训练并上传参数（先简化客户端选择过程，后期有时间再做得实际一点）
	void*          model;                          //实例化的模型类指针：用于保存上一轮聚合后的参数以及模型评估
	vector<vector<double> > data;
	vector<int> labels;

public:
	server(string Id, vector<string> Clt_id, int Global_epochs, int Local_epochs, vector<int> Batch_size, int K, void* Model, vector<vector<double> > Data, vector<int> Labels);
	bool FedLearning();
	vector<int>  select_clt(int clt_num);//使用随机数选择客户端进行训练,clt_nums是所选客户端个数。
	bool  para_fusion(vector<string>& clt);
	bool  evaluate();  //根据训练集评估当前模型（以分类问题为例，绘制roc，pr曲线等，求得混淆矩阵，精确率，查全率，查准率等）
};

#endif