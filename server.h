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
//���߳�ģ������
class server
{
private:
	string         id;                             //���߳�id
	SOCKET         s;
	SOCKET         c;
	vector<string> clt_id;                         //�洢�ͻ����̱߳�ʶ
	int            global_epochs;                  //�����ۺϴ��� �������ϴ�/�·�����
	int            local_epochs;                   //�ͻ���ģ��ѵ����������������磩
	vector<int>    batch_size;                     //�ͻ���ÿ��ѵ����ʹ�õ����ݼ���С��ʵ���п���ÿ���ͻ���ÿ��ѵ������С����ͬ,�����ۺ�ʱ��Ҫ�ò������м�Ȩ��
	int            k;                              //ÿ����ѡk���ͻ����߳̽���ѵ�����ϴ��������ȼ򻯿ͻ���ѡ����̣�������ʱ��������ʵ��һ�㣩
	void*          model;                          //ʵ������ģ����ָ�룺���ڱ�����һ�־ۺϺ�Ĳ����Լ�ģ������
	vector<vector<double> > data;
	vector<int> labels;

public:
	server(string Id, vector<string> Clt_id, int Global_epochs, int Local_epochs, vector<int> Batch_size, int K, void* Model, vector<vector<double> > Data, vector<int> Labels);
	bool FedLearning();
	vector<int>  select_clt(int clt_num);//ʹ�������ѡ��ͻ��˽���ѵ��,clt_nums����ѡ�ͻ��˸�����
	bool  para_fusion(vector<string>& clt);
	bool  evaluate();  //����ѵ����������ǰģ�ͣ��Է�������Ϊ��������roc��pr���ߵȣ���û������󣬾�ȷ�ʣ���ȫ�ʣ���׼�ʵȣ�
};

#endif