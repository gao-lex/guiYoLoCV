#include "guiYoLoCV.h"
#include <QtWidgets/QApplication>


int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	guiYoLoCV w;
	w.show();
	return a.exec();
}

