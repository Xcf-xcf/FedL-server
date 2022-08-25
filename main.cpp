#include <winsock2.h> // winsock2的头文件
#include <iostream>
#include<string>
#include<vector>
#include<sstream>
#include "server.h"
#include"data.h"
#include"model.h"

#pragma comment(lib, "ws2_32.lib")
using namespace std;

#pragma warning(disable:4996)
vector<string> clt_id({ "1","2","3","4","5","6","7","8","9","10" });
string svr_id = "127.0.0.1";
int local_epochs = 100;
int global_epochs = 100;
vector<int> batch_size({ 50,100,150,200,250,300,350,400,450,500 });
vector<vector<double> > Data;
vector<int> Labels;

int main()
{
	/*
	vector<vector<double> > x,y;
	vector<int> labels;
	read_in(x, y, "E:\\FedL\\server\\x.txt", "E:\\FedL\\server\\y.txt");
	y.resize(x.size());
	for (int i = 0; i < x.size(); i++)
	{
		y[i].resize(2);
		y[i][labels[i]] = 1;
	}
	x = normalize(x);
	vector< vector<vector<double> > > x_sect, y_sect;
	split_data(x_sect, y_sect, x, y, vector<double>({0.7,0.3}));
	//layer l1(3, 5, &sigmoid, &dsigmoid);
	//layer l2(3, 1, &Y, &Dy);
	layer l1(13, 20, &relu, &drelu);
	layer l2(20, 10, &relu, &drelu);
	//layer l3(20, 20, &relu, &drelu);
	layer l4(10, 1, &Y, &Dy);
	vector<layer> l;
	l.push_back(l1);
	l.push_back(l2);
	l.push_back(l4);
	//l.push_back(l3);
	//l.push_back(l4);
	BP bp(100, 16, 0.72, 0.9, 100, 0, l, &MSE, &dMSE, x_sect[0], y_sect[0]);
	bp.set_test(x_sect[1], y_sect[1]);
	bp.xavier_init();
	cout << "Training..." << endl;
	bp.train();
	vector<vector<vector<double> > > para = bp.get_para();
	//cout << para[0][0][0] << ' '<< para[0][0][1] << ' '<< para[0][0][2] << ' '<<para[0][0][3] << endl;
	cout << "Predicting..." << endl;
	vector<vector<double> > yy = bp.predict(x_sect[1]);
	vector<int> pred_labels = argmax(yy, 1);
	vector<int> true_labels = argmax(y_sect[1], 1);
	printf("准确率: %5.2lf%%\n", accuracy(true_labels, pred_labels) * 100);
	printf("查准率: %5.2lf%% %5.2lf%%\n", precision(true_labels, pred_labels)[0] * 100, precision(true_labels, pred_labels)[1] * 100);
	printf("查全率: %5.2lf%% %5.2lf%%\n", recall(true_labels, pred_labels)[0] * 100, recall(true_labels, pred_labels)[1] * 100);
	cout << "MSE: " << MSE(yy, y_sect[1]);
	*/
	
	WSADATA wd;
	if (WSAStartup(MAKEWORD(2, 2), &wd) != 0)
	{
		cout << "WSAStartup Error:" << WSAGetLastError() << endl;
		return 0;
	}
	read_in(Data, Labels, "E:\\FedL\\server\\data.txt", "E:\\FedL\\server\\labels.txt");
	
	cout << "Server Config:\n";
	cout << "Server IP: " << svr_id << endl;
	cout << "Local_Epochs: " << local_epochs << endl;
	cout << "Global_Epochs: " << global_epochs << endl;

	LDA model(Data[0].size());
	server svr(svr_id, clt_id, global_epochs, local_epochs, batch_size, 6, &model, Data, Labels);
	svr.FedLearning();

	// 清理winsock2的环境
	WSACleanup();

	return 0;
}