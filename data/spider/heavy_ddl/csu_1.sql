DROP DATABASE IF EXISTS csu_1;
CREATE DATABASE csu_1;
ALTER SESSION SET CURRENT_DATABASE = 'csu_1';
CREATE TABLE "Campuses" (
	"Id" INTEGER , 
	"Campus" TEXT, 
	"Location" TEXT, 
	"County" TEXT, 
	"Year" INTEGER 
);

CREATE TABLE "csu_fees" ( 
	"Campus" INTEGER , 
	"Year" INTEGER, 
	"CampusFee" INTEGER);

CREATE TABLE "degrees" ( 
	"Year" INTEGER,
	"Campus" INTEGER, 
	"Degrees" INTEGER);



CREATE TABLE "discipline_enrollments" ( 
	"Campus" INTEGER, 
	"Discipline" INTEGER, 
	"Year" INTEGER, 
	"Undergraduate" INTEGER, 
	"Graduate" INTEGER);



CREATE TABLE "enrollments" ( 
	"Campus" INTEGER, 
	"Year" INTEGER, 
	"TotalEnrollment_AY" INTEGER, 
	"FTE_AY" INTEGER);

CREATE TABLE "faculty" ( 
	"Campus" INTEGER, 
	"Year" INTEGER, 
	"Faculty" FLOAT);


