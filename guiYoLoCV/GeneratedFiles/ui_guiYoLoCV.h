/********************************************************************************
** Form generated from reading UI file 'guiYoLoCV.ui'
**
** Created by: Qt User Interface Compiler version 5.12.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_GUIYOLOCV_H
#define UI_GUIYOLOCV_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_guiYoLoCVClass
{
public:
    QWidget *centralWidget;
    QHBoxLayout *horizontalLayout;
    QLabel *videoLabel;
    QVBoxLayout *verticalLayout;
    QLabel *WarnLabel;
    QLabel *infoLabel;
    QPushButton *overloadAreaPushButton;

    void setupUi(QMainWindow *guiYoLoCVClass)
    {
        if (guiYoLoCVClass->objectName().isEmpty())
            guiYoLoCVClass->setObjectName(QString::fromUtf8("guiYoLoCVClass"));
        guiYoLoCVClass->resize(815, 490);
        centralWidget = new QWidget(guiYoLoCVClass);
        centralWidget->setObjectName(QString::fromUtf8("centralWidget"));
        QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(centralWidget->sizePolicy().hasHeightForWidth());
        centralWidget->setSizePolicy(sizePolicy);
        horizontalLayout = new QHBoxLayout(centralWidget);
        horizontalLayout->setSpacing(6);
        horizontalLayout->setContentsMargins(11, 11, 11, 11);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        videoLabel = new QLabel(centralWidget);
        videoLabel->setObjectName(QString::fromUtf8("videoLabel"));
        sizePolicy.setHeightForWidth(videoLabel->sizePolicy().hasHeightForWidth());
        videoLabel->setSizePolicy(sizePolicy);
        videoLabel->setAlignment(Qt::AlignCenter);

        horizontalLayout->addWidget(videoLabel);

        verticalLayout = new QVBoxLayout();
        verticalLayout->setSpacing(6);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        WarnLabel = new QLabel(centralWidget);
        WarnLabel->setObjectName(QString::fromUtf8("WarnLabel"));
        sizePolicy.setHeightForWidth(WarnLabel->sizePolicy().hasHeightForWidth());
        WarnLabel->setSizePolicy(sizePolicy);
        WarnLabel->setAlignment(Qt::AlignCenter);

        verticalLayout->addWidget(WarnLabel);

        infoLabel = new QLabel(centralWidget);
        infoLabel->setObjectName(QString::fromUtf8("infoLabel"));
        sizePolicy.setHeightForWidth(infoLabel->sizePolicy().hasHeightForWidth());
        infoLabel->setSizePolicy(sizePolicy);
        infoLabel->setAlignment(Qt::AlignCenter);

        verticalLayout->addWidget(infoLabel);

        overloadAreaPushButton = new QPushButton(centralWidget);
        overloadAreaPushButton->setObjectName(QString::fromUtf8("overloadAreaPushButton"));

        verticalLayout->addWidget(overloadAreaPushButton);


        horizontalLayout->addLayout(verticalLayout);

        guiYoLoCVClass->setCentralWidget(centralWidget);

        retranslateUi(guiYoLoCVClass);

        QMetaObject::connectSlotsByName(guiYoLoCVClass);
    } // setupUi

    void retranslateUi(QMainWindow *guiYoLoCVClass)
    {
        guiYoLoCVClass->setWindowTitle(QApplication::translate("guiYoLoCVClass", "guiYoLoCV", nullptr));
        videoLabel->setText(QApplication::translate("guiYoLoCVClass", "\346\255\243\345\234\250\345\212\240\350\275\275\345\233\276\345\203\217...", nullptr));
        WarnLabel->setText(QApplication::translate("guiYoLoCVClass", "\350\255\246\345\221\212\344\277\241\346\201\257", nullptr));
        infoLabel->setText(QApplication::translate("guiYoLoCVClass", "\346\243\200\346\265\213ing", nullptr));
        overloadAreaPushButton->setText(QApplication::translate("guiYoLoCVClass", "\351\207\215\350\275\275\351\231\220\345\210\266\345\214\272\345\237\237", nullptr));
    } // retranslateUi

};

namespace Ui {
    class guiYoLoCVClass: public Ui_guiYoLoCVClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_GUIYOLOCV_H
