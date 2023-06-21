DROP DATABASE IF EXISTS music_1;
CREATE DATABASE music_1;
ALTER SESSION SET CURRENT_DATABASE = 'music_1';

create table genre(
	g_name text not null,
	rating text,
	most_popular_in text);

create table artist(
	artist_name text not null,
	country text,
	gender text,
	preferred_genre text);

create table files(
	f_id decimal(10, 0) not null,
	artist_name text,
	file_size text,
	duration text,
	formats text);


create table song(
	song_name text,
	artist_name text,
	country text,
	f_id decimal(10, 0),
    	genre_is text,
	rating decimal(10, 0),
	languages text,
	releasedate Date, 
	resolution decimal(10, 0) not null);


insert into genre(g_name,rating,most_popular_in) values ('tagore','8','Bangladesh');
insert into genre values ('nazrul','7','Bangladesh');
insert into genre values ('folk','9','Sylhet,Chittagong,Kustia');
insert into genre values ('modern','8','Bangladesh');
insert into genre values ('blues','7','Canada');
insert into genre values ('pop','9','America');


insert into artist(artist_name,country,gender,preferred_genre) values('Shrikanta','India','Male','tagore');
insert into artist values('Prity','Bangladesh','Female','nazrul');
insert into artist values('Farida','Bangladesh','Female','folk');
insert into artist values('Topu','India','Female','modern');
insert into artist values('Enrique','USA','Male','blues');
insert into artist values('Michel','UK','Male','pop');


insert into files(f_id,artist_name,file_size,duration,formats) values (1,'Shrikanta','3.78 MB','3:45','mp4');
insert into files values (2,'Prity','4.12 MB','2:56','mp3');
insert into files values (3,'Farida','3.69 MB','4:12','mp4');
insert into files values (4,'Enrique','4.58 MB','5:23','mp4');
insert into files values (5,'Michel','5.10 MB','4:34','mp3');
insert into files values (6,'Topu','4.10 MB','4:30','mp4');

insert into song(song_name,artist_name,country,f_id,genre_is,rating,languages,releasedate,resolution) values ('Tumi robe nirobe','Shrikanta','India','1','tagore','8','bangla','2001-08-28',1080);
insert into song values ('Shukno patar nupur pae','Prity','Bangladesh','2','nazrul','5','bangla','1997-09-21',512);
insert into song values ('Ami opar hoye','Farida','Bangladesh','3','folk','7','bangla','2001-04-07',320);
insert into song values ('My love','Enrique','USA','4','blues','6','english','2007-01-24',1080);
insert into song values ('Just beat it','Michel','UK','5','pop','8','english','2002-03-17',720);
insert into song values ('Aj ei akash','Topu','India','6','modern','10','bangla','2004-03-27',320);
