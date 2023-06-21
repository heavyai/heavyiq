DROP DATABASE IF EXISTS inn_1;
CREATE DATABASE inn_1;
ALTER SESSION SET CURRENT_DATABASE = 'inn_1';
CREATE TABLE "Rooms" ( 
	"RoomId" TEXT ,
	"roomName" TEXT, 
	"beds" INTEGER, 
	"bedType" TEXT, 
	"maxOccupancy" INTEGER, 
	"basePrice" INTEGER, 
	"decor" TEXT

);

CREATE TABLE "Reservations" ( 
	"Code" INTEGER , 
	"Room" TEXT, 
	"CheckIn" TEXT, 
	"CheckOut" TEXT, 
	"Rate" FLOAT, 
	"LastName" TEXT, 
	"FirstName" TEXT, 
	"Adults" INTEGER, 
	"Kids" INTEGER);

