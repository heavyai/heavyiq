DROP DATABASE IF EXISTS yelp;
CREATE DATABASE yelp;
ALTER SESSION SET CURRENT_DATABASE = 'yelp';
CREATE TABLE "business" (
"bid" int,
"business_id" text,
"name" text,
"full_address" text,
"city" text,
"latitude" text,
"longitude" text,
"review_count" int,
"is_open" int,
"rating" FLOAT,
"state" text);
CREATE TABLE "category" (
"id" int,
"business_id" text,
"category_name" text);
CREATE TABLE "user_" (
"uid" int,
"user_id" text,
"name" text);
CREATE TABLE "checkin" (
"cid" int,
"business_id" text,
"count" int,
"day" text);

CREATE TABLE "neighbourhood" (
"id" int,
"business_id" text,
"neighbourhood_name" text);

CREATE TABLE "review" (
"rid" int,
"business_id" text,
"user_id" text,
"rating" FLOAT,
"text" text,
"year" int,
"month" text);
CREATE TABLE "tip" (
"tip_id" int,
"business_id" text,
"text" text,
"user_id" text,
"likes" int,
"year" int,
"month" text);
