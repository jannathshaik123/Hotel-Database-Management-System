show databases;
create database hotel_management;
use hotel_management;
show tables;
create table GUEST(G_ID VARCHAR(5) PRIMARY KEY,
					G_FNAME VARCHAR(20),
					G_LNAME VARCHAR(20),
                    AGE INT,
                    GENDER VARCHAR(6),
                    NATIONALITY VARCHAR(25),
                    PHONE_NUMBER VARCHAR(15), 
                    EMAIL VARCHAR(100),
                    ADDRESS VARCHAR(30),
					PASSPORT_NUMBER VARCHAR(10));
select * from guest;
create table MEMBERS(G_ID varchar(5),
					M_NAME VARCHAR(30),	
					PASSPORT_NUMBER VARCHAR(10),
                    FOREIGN KEY(G_ID) REFERENCES GUEST(G_ID));

CREATE TABLE STAFF(S_ID VARCHAR(5) PRIMARY KEY,
					S_FNAME VARCHAR(20),
                    S_LNAME VARCHAR(20),
                    SALARY INT,
                    GENDER VARCHAR(6),
                    NATIONALITY VARCHAR(25),
                    ADDRESS VARCHAR(40),
                    EMAIL VARCHAR(100),
                    DEPT_NAME VARCHAR(15),
                    POSITION VARCHAR(20));
desc staff;
CREATE TABLE LOGIN(S_ID VARCHAR(5),
					USERNAME VARCHAR(20),
                    PASSWORD VARCHAR(20),
                    FOREIGN KEY(S_ID) REFERENCES STAFF(S_ID));
CREATE TABLE ROOM (R_NO VARCHAR(5) PRIMARY KEY,
					TYPE VARCHAR(10),
                    PRICE INT,
                    OCCUPANCY INT,
                    STATUS varchar(15),
                    DESCRIPTION VARCHAR(100),
                    check(STATUS IN('VACANT','OCCUPIED')));
select * from room;
create table BOOKING(B_ID VARCHAR(5) PRIMARY KEY,
					G_ID VARCHAR(5),
                    R_NO VARCHAR(5),
                    B_DATE TIMESTAMP,
                    CHECK_IN DATE,
                    CHECK_OUT DATE,
                    DURATION int AS (datediff(CHECK_OUT,CHECK_IN)),
                    FOREIGN KEY(R_NO) REFERENCES ROOM(R_NO),
                    foreign key(G_ID) REFERENCES GUEST(G_ID));

create table payment(P_ID VARCHAR(5) PRIMARY KEY,
					B_ID VARCHAR(5),
                    P_DATE DATE,
                    METHOD VARCHAR(10),
                    R_PRICE INT,
                    TIP INT,
                    FOREIGN KEY(B_ID) REFERENCES BOOKING(B_ID),
                    EXTRA_CHARGES INT);
select * from login;
INSERT INTO GUEST(G_ID) VALUE('G100');
INSERT INTO ROOM VALUES('R101','SMALL',200,1,'VACANT','HELLO');
insert into booking (B_ID,G_ID,R_NO,B_DATE,CHECK_IN,CHECK_OUT) values ('B101','G100','R100','2005-7-27 09:00:30.75','2023-04-16','2023-04-19');
insert into booking (B_ID,G_ID,R_NO,B_DATE,CHECK_IN,CHECK_OUT) values ('B102','G101','R102','2005-7-27 09:00:30.75','2023-04-14','2023-04-17');
insert into booking (B_ID,G_ID,R_NO,B_DATE,CHECK_IN,CHECK_OUT) values ('B103','G111','R105','2005-7-27 09:00:30.75','2023-04-18','2023-04-20');
delete from booking where b_id='B101';

/*Function used to generate random P_ID for payements table*/
delimiter !
CREATE FUNCTION generate_id() RETURNS VARCHAR(4)
BEGIN
    DECLARE chars VARCHAR(10) DEFAULT '0123456789';
    DECLARE id VARCHAR(4);
    SET id = CONCAT('P', FLOOR(RAND() * 1000));
    RETURN id;
END!
delimiter ;
SET GLOBAL log_bin_trust_function_creators = 1;

/*Trigger to insert into payment table after inserting into bookings*/

delimiter !
create trigger setprice
after INSERT ON BOOKING
for each row
begin
	DECLARE RP int;
    DECLARE DUR INT;
    DECLARE pid varchar(5);
	set RP=(select room.price from room NATURAL JOIN BOOKING where ROOM.R_NO=NEW.R_NO);
    SET DUR=(SELECT DURATION FROM BOOKING WHERE B_ID=NEW.B_ID);
    set pid=(SELECT generate_id());
	insert into payment(P_ID,B_ID,R_PRICE,TIP,EXTRA_CHARGES) 
    VALUES (pid,NEW.B_ID,(DUR*RP),0,0);
end !
delimiter ;

/* Function for total price at checkout*/

delimiter !
create function calc_tot(PID VARCHAR(5)) returns int
begin
	declare total int;
    set total=((select R_PRICE FROM PAYMENT WHERE P_ID=PID)+(select TIP FROM PAYMENT WHERE P_ID=PID)+(select EXTRA_CHARGES FROM PAYMENT WHERE P_ID=PID));
    RETURN total;
END !
DELIMITER ;
select calc_tot('P224');

/* Trigger for chaging room status to occupied after booking*/

DELIMITER !
CREATE TRIGGER UPDATE_STATUS
AFTER INSERT ON BOOKING 
FOR EACH ROW 
BEGIN
	update ROOM SET STATUS="OCCUPIED" WHERE R_NO=NEW.R_NO;
END!
DELIMITER ;

/*Trigger to change room status back to vacant after current date*/

delimiter !
create trigger UPDATE_STATUS_2
AFTER INSERT ON BOOKING
FOR EACH ROW
BEGIN 
	declare d date;
    SET d=(select CURRENT_DATE());
	update ROOM SET STATUS="VACANT" WHERE R_NO IN (SELECT R_NO FROM BOOKING WHERE CHECK_OUT< d);
END!
DELIMITER ;
update login set password='shangrinah.2023' where s_id is not null;



	