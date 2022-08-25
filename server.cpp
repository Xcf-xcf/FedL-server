#include<iostream>
#include<string>
#include<sstream>
#include"server.h"
#include <winsock2.h>
#pragma comment(lib, "ws2_32.lib")
#pragma warning(disable:4996)
using namespace std;

server::server(string Id, vector<string> Clt_id, int Global_epochs, int Local_epochs, vector<int> Batch_size, int K, void* Model, vector<vector<double> > Data, vector<int> Labels)
{
	id = Id;
	clt_id.assign(Clt_id.begin(), Clt_id.end());
	global_epochs = Global_epochs;
	local_epochs = Local_epochs;
	batch_size.assign(Batch_size.begin(), Batch_size.end());
	k = K;
	model = Model;
	data.assign(Data.begin(), Data.end());
	labels.assign(Labels.begin(), Labels.end());
	// 1. 创建流式套接字
	s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
	if (s == INVALID_SOCKET)
	{
		cout << "socket error:" << WSAGetLastError() << endl;
	}
	// 2. 绑定端口和ip
	sockaddr_in addr;
	memset(&addr, 0, sizeof(sockaddr_in));
	addr.sin_family = AF_INET;
	addr.sin_port = htons(8000);
	addr.sin_addr.s_addr = inet_addr(id.c_str());

	int len = sizeof(sockaddr_in);
	if (bind(s, (SOCKADDR*)&addr, len) == SOCKET_ERROR)
	{
		cout << "bind Error:" << WSAGetLastError() << endl;
	}
	// 3. 监听
	listen(s, MaxClientNum);
	// 接收客户端的连接
	cout << "Listenning..." << endl;
	sockaddr_in addrClient;
	len = sizeof(sockaddr_in);
	do
	{
		c = accept(s, (SOCKADDR*)&addrClient, &len);
	} while (c == INVALID_SOCKET);
	cout << "Connected Clients!" << endl;
}

bool server::FedLearning()
{
	int cnt = 0;
	cout << "------------Epoch " << cnt + 1 << "------------" << endl;
	while (cnt < global_epochs)
	{
		char msg[1];
		recv(c, msg, 1, 0);
		if (msg[0] == 'S')
		{
			char buf[BufSize] = { 0 };
			recv(c, buf, BufSize, 0);
			string str(buf);
			stringstream ss(str);
			str = "";
			vector<string> clt;
			while (getline(ss, str))
			{
				clt.push_back(str.substr(str.find(':')+1));
			}
			cnt++;
			cout << "Received Local Parameters!" << endl;
			para_fusion(clt);
			cout << "Parameter Aggregation Done!" << endl;
			cout << endl;
			cout << "Result: " << endl;
			evaluate();
			cout << endl;
			cout << "----------------------------------" << endl;
		}
		else if (msg[0] == 'R')
		{
			cout << "------------Epoch " << cnt + 1 << "------------" << endl;
			char buf[BufSize] = { 0 };
			sprintf(buf, model->para_to_str(model->get_para()).c_str());
			send(c, buf, BufSize, 0);
			cout << "Parameters Has Been Delivered!" << endl;
		}
	}
	return true;
}

vector<int>  server::select_clt(int clt_num)
{
	vector<int> clt;
	int tot = clt_id.size();
	for (int i = 0; i < tot; i++)
	{
		clt.push_back(i);
	}
	srand(time(0));
	random_shuffle(clt.begin(), clt.end());
	vector<int> index;
	index.assign(clt.begin(), clt.begin() + clt_num);
	return index;
}

bool  server::para_fusion(vector<string>& clt)
{
	vector<int> index = select_clt(k);
	//cout << clt[index[0]] << endl;
	vector<double> para = model->str_to_para(clt[index[0]]);
	int data_size = 0;
	for (int i = 0; i < batch_size.size(); i++)
	{
		//cout << clt[i] << endl;
		data_size += batch_size[i];
	}
	for (int j = 0; j < para.size(); j++)
	{
		para[j] *= (1.0 * batch_size[0] / data_size);
	}
	for (int i = 1; i < index.size(); i++)
	{
		for (int j = 0; j < para.size(); j++)
		{
			vector<double> p = model->str_to_para(clt[index[i]]);
			para[j] += p[j] * (1.0 * batch_size[i] / data_size);
		}
	}
	//cout << clt.size() << endl;
	//cout << para.size() << endl;
	model->set_para(para);
	return true;
}

bool  server::evaluate()
{
	vector<int> pred_labels = model->predict(data);
	//cout << 1 << endl;
	printf("准确率: %5.2lf%%\n", accuracy(labels, pred_labels) * 100);
	printf("查准率: %5.2lf%% %5.2lf%%\n", precision(labels, pred_labels)[0] * 100, precision(labels, pred_labels)[1] * 100);
	printf("查全率: %5.2lf%% %5.2lf%%\n", recall(labels, pred_labels)[0] * 100, recall(labels, pred_labels)[1] * 100);
	FILE* file = fopen("result.txt", "a+");
	stringstream ss;
	ss << setprecision(15) << accuracy(labels, pred_labels);
	ss << "\n";
	fputs(ss.str().c_str(), file);
	fclose(file);
	return true;
}