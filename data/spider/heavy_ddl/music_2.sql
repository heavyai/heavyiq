DROP DATABASE IF EXISTS music_2;
CREATE DATABASE music_2;
ALTER SESSION SET CURRENT_DATABASE = 'music_2';


CREATE TABLE "Songs" ( 
	"SongId" INTEGER , 
	"Title" TEXT 
);
CREATE TABLE "Albums" ( 
	"AId" INTEGER , 
	"Title" TEXT, 
	"Year" INTEGER, 
	"Label" TEXT, 
	"Type" TEXT );
CREATE TABLE "Band" ( 
	"Id" INTEGER , 
	"Firstname" TEXT, 
	"Lastname" TEXT );
CREATE TABLE "Instruments" ( 
	"SongId" INTEGER, 
	"BandmateId" INTEGER, 
	"Instrument" TEXT );
CREATE TABLE "Performance" ( 
	"SongId" INTEGER, 
	"Bandmate" INTEGER, 
	"StagePosition" TEXT);
CREATE TABLE "Tracklists" ( 
	"AlbumId" INTEGER, 
	"Position" INTEGER, 
	"SongId" INTEGER );
CREATE TABLE "Vocals" ( 
	"SongId" INTEGER, 
	"Bandmate" INTEGER, 
	"Type" TEXT);

