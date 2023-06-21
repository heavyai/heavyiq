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
	"Country" TEXT);


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
	"MPG" TEXT, 
	"Cylinders" INTEGER, 
	"Edispl" FLOAT, 
	"Horsepower" TEXT, 
	"Weight" INTEGER, 
	"Accelerate" FLOAT, 
	"Year" INTEGER);


