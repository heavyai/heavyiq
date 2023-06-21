DROP DATABASE IF EXISTS wine_1;
CREATE DATABASE wine_1;
ALTER SESSION SET CURRENT_DATABASE = 'wine_1';
CREATE TABLE "grapes" ( 
	"ID" INTEGER , 
	"Grape" TEXT, 
	"Color" TEXT 
);

CREATE TABLE "appellations" ( 
	"No" INTEGER , 
	"Appelation" TEXT, 
	"County" TEXT, 
	"State" TEXT, 
	"Area" TEXT, 
	"isAVA" TEXT
);

CREATE TABLE "wine" ( 
	"No" INTEGER, 
	"Grape" TEXT, 
	"Winery" TEXT, 
	"Appelation" TEXT, 
	"State" TEXT, 
	"Name" TEXT, 
	"Year" INTEGER, 
	"Price" INTEGER, 
	"Score" INTEGER, 
	"Cases" INTEGER, 
	"Drink" TEXT);
