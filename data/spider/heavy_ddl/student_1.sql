DROP DATABASE IF EXISTS student_1;
CREATE DATABASE student_1;
ALTER SESSION SET CURRENT_DATABASE = 'student_1';
CREATE TABLE "list" ( 
	"LastName" TEXT, 
	"FirstName" TEXT, 
	"Grade" INTEGER, 
	"Classroom" INTEGER);
CREATE TABLE "teachers" ( 
	"LastName" TEXT, 
	"FirstName" TEXT, 
	"Classroom" INTEGER);
