DROP DATABASE IF EXISTS car_1;
CREATE DATABASE car_1;
ALTER SESSION SET CURRENT_DATABASE = 'car_1';
CREATE TABLE "continents" ( 
	"ContId" INTEGER , 
	"Continent" TEXT 
);

CREATE TABLE "countries" (
	"CountryId" INTEGER , 
	"CountryName" TEXT, 
	"Continent" INTEGER);


CREATE TABLE "car_makers" ( 
	"Id" INTEGER , 
	"Maker" TEXT, 
	"FullName" TEXT, 
	"CountryId" INT);


CREATE TABLE "model_list" ( 
	"ModelId" INTEGER , 
	"Maker" INTEGER, 
	"Model" TEXT);



CREATE TABLE "car_names" ( 
	"MakeId" INTEGER , 
	"Model" TEXT, 
	"Make" TEXT);

CREATE TABLE "cars_data" (
	"Id" INTEGER , 
	"MPG" FLOAT, 
	"Cylinders" INTEGER, 
	"Edispl" FLOAT, 
	"Horsepower" INTEGER, 
	"Weight" INTEGER, 
	"Accelerate" FLOAT, 
	"Year" INTEGER);

INSERT INTO car_names VALUES (1,'chevrolet chevelle malibu','chevrolet'); 
INSERT INTO cars_data VALUES (1,18,8,307,130,3504,12,70);


