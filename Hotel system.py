from datetime import date
import tkinter as tk
from tkinter import *
from tkinter import messagebox
from PIL import Image,ImageTk
from tkinter import ttk
from tkcalendar import Calendar
import tkcalendar as tkc
import mysql.connector
from datetime import datetime


mydb=mysql.connector.connect(host="localhost", user="root", password="3475", database="hotel_management")

def main():
    def staff_login():       #Staff login window
        sroot=tk.Toplevel(root)
        sroot.title("Staff Login")
        sroot.geometry("1920x1080")
        root.withdraw()
        vertical=tk.Frame(sroot,bg='#A2A2CD',height=1080,width=170) 
        vertical.place(x=0,y=0)
        heading=tk.Frame(sroot,bg='#A2A2CD',height=170,width=1920)
        heading.pack()
        main_logo=ImageTk.PhotoImage((Image.open("Images\logo.png")).resize((100,100)))
        picture=tk.Label(heading,image=main_logo,bg='#A2A2CD')
        picture.photo=main_logo
        picture.pack(side="left",ipadx=30)
        title=tk.Label(heading,text="Shangri-Nah Hotel",font=('Helvetica',36,'bold'),bg='#A2A2CD')
        title.pack(padx=475,pady=30)
        frame=tk.Frame(sroot,bg='#FAFAEB',height=700,width=1360)
        frame.place(x=171,y=121)
        def login():
            def guest_view(): #Guest view button function
                gv_frame=tk.Frame(sroot,height=700,width=1360)
                gv_frame.place(x=171,y=121)
                label1=tk.Label(gv_frame,text="GUEST ID:",font=('Arial',18))
                label1.pack(side='left',padx=(350,20),pady=(300,325))
                gid_text=tk.Text(gv_frame,font=('Arial',18),height=1,width=25)
                gid_text.pack(side='left',padx=25,pady=(300,325))
                def takegid(): #check button function                      
                    bno=gid_text.get(1.0,"end-1c")                                   

                    cur8=mydb.cursor()
                    q8="SELECT * FROM GUEST WHERE G_ID=%s"                                 
                    cur8.execute(q8, (bno, ))
                    guest=cur8.fetchall()
                    g_fname=guest[0][1]                                       
                    g_lname=guest[0][2]
                    g_age=str(guest[0][3])
                    g_gender=guest[0][4]
                    g_nation=guest[0][5]
                    g_phone=str(guest[0][6])
                    g_email=guest[0][7]
                    g_add=guest[0][8]
                    g_passno=guest[0][9]
                    cur8.close()

                    messagebox.showinfo(title="Guest details", message='Guest name: '+g_fname+" "+g_lname+"\n"+"Age: "+g_age \
                                        +"\n"+"Gender: "+g_gender+"\n"+"Nationality: "+g_nation+"\n"+"Phone number: "+g_phone+"\n"\
                                        +"Email: "+g_email+"\n"+"Address: "+g_add+"\n"+"Passport number: "+g_passno)
                    
                    
                check=tk.Button(gv_frame,text="Check",font=('Arial',16),width=10,command=takegid)
                check.pack(padx=20,pady=(300,325),side='left')

                exit=tk.Button(gv_frame,text="Exit",font=('Arial',16),width=10, command=smain)
                exit.pack(padx=(20,1000),pady=(300,325), side='left')
                
            def update_guest():  #update guest button function
                uv_frame=tk.Frame(sroot,bg='#FAFAEB',height=700,width=1360)
                uv_frame.place(x=171,y=121)
                gid_label=tk.Label(uv_frame,text="Guest ID:",font=('Arial',18))
                gid_label.grid(row=0,column=0,padx=(180,5),pady=(70,30))
                gid_text=tk.Text(uv_frame,font=('Arial',16),height=1,width=25)
                gid_text.grid(row=0,column=1,padx=(10,20),pady=(70,30))

                fname_label=tk.Label(uv_frame,text="First Name:",font=('Arial',18))
                fname_label.grid(row=1,column=0,padx=(180,5),pady=(70,30))
                fname_text=tk.Text(uv_frame,font=('Arial',16),height=1,width=25)
                fname_text.grid(row=1,column=1,padx=(10,20),pady=(70,30))
            
                lname_label=tk.Label(uv_frame,text="Last Name:",font=('Arial',18))
                lname_label.grid(row=2,column=0,padx=(180,5),pady=(70,30))
                lname_text=tk.Text(uv_frame,font=('Arial',16),height=1,width=25)
                lname_text.grid(row=2,column=1,padx=(10,20),pady=(70,30))
            
                age_label=tk.Label(uv_frame,text="Age:",font=('Arial',18))   
                age_label.grid(row=3,column=0,padx=(250,5),pady=(70,30))
                age_text=tk.Text(uv_frame,font=('Arial',16),height=1,width=25)
                age_text.grid(row=3,column=1,padx=(10,20),pady=(70,30))
            
            
                email_label=tk.Label(uv_frame,text="Email:",font=('Arial',18))
                email_label.grid(row=4,column=0,padx=(230,5),pady=(70,30))
                email_text=tk.Text(uv_frame,font=('Arial',16),height=1,width=25)
                email_text.grid(row=4,column=1,padx=(10,20),pady=(70,30))
            
                phone_label=tk.Label(uv_frame,text="Phone:",font=('Arial',18))
                phone_label.grid(row=0,column=3,padx=(220,5),pady=(70,30))
                phone_text=tk.Text(uv_frame,font=('Arial',16),height=1,width=30)
                phone_text.grid(row=0,column=4,padx=(10,120),pady=(70,30))
            
                address_label=tk.Label(uv_frame,text="Address:",font=('Arial',18))
                address_label.grid(row=1,column=3,padx=(200,5),pady=(70,30))
                address_text=tk.Text(uv_frame,font=('Arial',16),height=1,width=30)
                address_text.grid(row=1,column=4,padx=(10,120),pady=(70,30))

                passport_label=tk.Label(uv_frame,text="Passport No:",font=('Arial',18))
                passport_label.grid(row=2,column=3,padx=(160,5),pady=(70,30))
                passport_text=tk.Text(uv_frame,font=('Arial',16),height=1,width=30)
                passport_text.grid(row=2,column=4,padx=(10,120),pady=(70,30))

                country_label=tk.Label(uv_frame,text="Nationality:",font=('Arial',18))
                country_label.grid(row=3,column=3,padx=(180,10),pady=(70,30))
                country=StringVar()
                country_drop=ttk.Combobox(uv_frame,textvariable=country,width=25,font=('Arial',16))
                country_drop.grid(row=3,column=4,padx=(10,120),pady=(70,30))
                country_drop['values']=['Afghanistan', 'Aland Islands', 'Albania', 'Algeria', 'American Samoa', 'Andorra', 'Angola', 'Anguilla', 'Antarctica', 'Antigua and Barbuda', 'Argentina', 'Armenia', 'Aruba', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin', 'Bermuda', 'Bhutan', 'Bolivia, Plurinational State of', 'Bonaire, Sint Eustatius and Saba', 'Bosnia and Herzegovina', 'Botswana', 'Bouvet Island', 'Brazil', 'British Indian Ocean Territory', 'Brunei Darussalam', 'Bulgaria', 'Burkina Faso', 'Burundi', 'Cambodia', 'Cameroon', 'Canada', 'Cape Verde', 'Cayman Islands', 'Central African Republic', 'Chad', 'Chile', 'China', 'Christmas Island', 'Cocos (Keeling) Islands', 'Colombia', 'Comoros', 'Congo', 'Congo, The Democratic Republic of the', 'Cook Islands', 'Costa Rica', "Côte d'Ivoire", 'Croatia', 'Cuba', 'Curaçao', 'Cyprus', 'Czech Republic', 'Denmark', 'Djibouti', 'Dominica', 'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Estonia', 'Ethiopia', 'Falkland Islands (Malvinas)', 'Faroe Islands', 'Fiji', 'Finland', 'France', 'French Guiana', 'French Polynesia', 'French Southern Territories', 'Gabon', 'Gambia', 'Georgia', 'Germany', 'Ghana', 'Gibraltar', 'Greece', 'Greenland', 'Grenada', 'Guadeloupe', 'Guam', 'Guatemala', 'Guernsey', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti', 'Heard Island and McDonald Islands', 'Holy See (Vatican City State)', 'Honduras', 'Hong Kong', 'Hungary', 'Iceland', 'India', 'Indonesia', 'Iran, Islamic Republic of', 'Iraq', 'Ireland', 'Isle of Man', 'Israel', 'Italy', 'Jamaica', 'Japan', 'Jersey', 'Jordan', 'Kazakhstan', 'Kenya', 'Kiribati', "Korea, Democratic People's Republic of", 'Korea, Republic of', 'Kuwait', 'Kyrgyzstan', "Lao People's Democratic Republic", 'Latvia', 'Lebanon', 'Lesotho', 'Liberia', 'Libya', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Macao', 'Macedonia, Republic of', 'Madagascar', 'Malawi', 'Malaysia', 'Maldives', 'Mali', 'Malta', 'Marshall Islands', 'Martinique', 'Mauritania', 'Mauritius', 'Mayotte', 'Mexico', 'Micronesia, Federated States of', 'Moldova, Republic of', 'Monaco', 'Mongolia', 'Montenegro', 'Montserrat', 'Morocco', 'Mozambique', 'Myanmar', 'Namibia', 'Nauru', 'Nepal', 'Netherlands', 'New Caledonia', 'New Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'Niue', 'Norfolk Island', 'Northern Mariana Islands', 'Norway', 'Oman', 'Pakistan', 'Palau', 'Palestinian Territory, Occupied', 'Panama', 'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines', 'Pitcairn', 'Poland', 'Portugal', 'Puerto Rico', 'Qatar', 'Réunion', 'Romania', 'Russian Federation', 'Rwanda', 'Saint Barthélemy', 'Saint Helena, Ascension and Tristan da Cunha', 'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Martin (French part)', 'Saint Pierre and Miquelon', 'Saint Vincent and the Grenadines', 'Samoa', 'San Marino', 'Sao Tome and Principe', 'Saudi Arabia', 'Senegal', 'Serbia', 'Seychelles', 'Sierra Leone', 'Singapore', 'Sint Maarten (Dutch part)', 'Slovakia', 'Slovenia', 'Solomon Islands', 'Somalia', 'South Africa', 'South Georgia and the South Sandwich Islands', 'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'South Sudan', 'Svalbard and Jan Mayen', 'Swaziland', 'Sweden', 'Switzerland', 'Syrian Arab Republic', 'Taiwan, Province of China', 'Tajikistan', 'Tanzania, United Republic of', 'Thailand', 'Timor-Leste', 'Togo', 'Tokelau', 'Tonga', 'Trinidad and Tobago', 'Tunisia', 'Turkey', 'Turkmenistan', 'Turks and Caicos Islands', 'Tuvalu', 'Uganda', 'Ukraine', 'United Arab Emirates', 'United Kingdom', 'United States', 'United States Minor Outlying Islands', 'Uruguay', 'Uzbekistan', 'Vanuatu', 'Venezuela, Bolivarian Republic of', 'Viet Nam', 'Virgin Islands, British', 'Virgin Islands, U.S.', 'Wallis and Futuna', 'Yemen', 'Zambia', 'Zimbabwe']
                
                def submit(): #submit button for guest update
                    g_id=gid_text.get(1.0,"end-1c")
                    fname=fname_text.get(1.0,"end-1c")
                    lname=lname_text.get(1.0,"end-1c")
                    age=age_text.get(1.0,"end-1c")
                    address=address_text.get(1.0,"end-1c")
                    email=email_text.get(1.0,"end-1c")
                    phone=phone_text.get(1.0,"end-1c")
                    passno=passport_text.get(1.0,"end-1c")
                    nationality=country.get()

                    cur9=mydb.cursor()  
                    cur9.execute("SELECT gender FROM GUEST WHERE G_ID=%s", (g_id,))
                    gender=cur9.fetchall()
                    cur9.close()
                    cur9=mydb.cursor()
                    q9="UPDATE GUEST SET G_FNAME=%s, G_LNAME=%s, AGE=%s, GENDER=%s, NATIONALITY=%s, PHONE_NUMBER=%s, EMAIL=%s, ADDRESS=%s, PASSPORT_NUMBER=%s WHERE G_ID=%s"
                    cur9.execute(q9, (fname, lname, age, gender[0][0], nationality, phone, email, address, passno, g_id))
                    mydb.commit()
                    cur9.close()

                    messagebox.showinfo(title="Updates",message="Updated")
                    smain()
                Submit_button=tk.Button(uv_frame,text="Update",command=submit,font=('Arial',18))
                Submit_button.grid(row=4,column=3,padx=(180,10),pady=(70,30))

            def checkbook(): # Display booking details 
                c_frame=tk.Frame(sroot,height=700,width=1360)
                c_frame.place(x=171,y=121)
                label1=tk.Label(c_frame,text="BOOKING NUMBER:",font=('Arial',18))
                label1.pack(side='left',padx=(350,20),pady=(300,325))
                bookingno=tk.Text(c_frame,font=('Arial',18),height=1,width=25)
                bookingno.pack(side='left',padx=25,pady=(300,325))
                def takebno(): #Check Button Function
                    bno=bookingno.get(1.0,"end-1c")
                    booking_cursor=mydb.cursor()
                    booking_cursor.execute("SELECT * FROM BOOKING WHERE B_ID=%s",(bno,))
                    booking_details=booking_cursor.fetchone()
                    booking_cursor.close()

                    ##getting booking details, make display page 
                    booking_number=booking_details[0]
                    guest_id=booking_details[1]
                    room_no=booking_details[2]
                    booking_date=str(booking_details[3])
                    checkin_date=str(booking_details[4])
                    checkout_date=str(booking_details[5])
                    duration_of_stay=str(booking_details[6])

                    messagebox.showinfo(title="Booking details for "+booking_number, message='Guest id: '+guest_id+"\n"+"Room number: "+room_no \
                                        +"\n"+"Booking Date: "+booking_date+"\n"+"Check-in Date: "+checkin_date+"\n"+"Check-out Date: "+checkout_date+"\n"\
                                        +"Duration of Stay: "+duration_of_stay)



                check=tk.Button(c_frame,text="Check",font=('Arial',16),width=10,command=takebno)
                check.pack(padx=20, pady=(300,325),side='left')

                exit=tk.Button(c_frame,text="Exit",font=('Arial',16),width=10,command=smain)
                exit.pack(padx=(20,1000),pady=(300,325),side='left')

                                
            def payment(): #Payment button Functioning 
                def save(): #function for saving payment details
                    bno=bid_text.get(1.0,"end-1c")
                    tip=int(tip_text.get(1.0,"end-1c"))
                    method=meth.get()
                    extra=int(extra_text.get(1.0,"end-1c"))
                    #display the values including total calling the function written in sql.

                    payment=date.today()
                    payment_date=str(payment.year)+'-'+str(payment.month)+'-'+str(payment.day)


                    payment_cursor=mydb.cursor()
                    payment_cursor.execute("UPDATE payment SET METHOD=%s,TIP= %s, EXTRA_CHARGES=%s, P_DATE=%s WHERE B_ID= %s",(method,tip,extra,payment_date,bno))
                    mydb.commit()

                    cur10=mydb.cursor()
                    cur10.execute("SELECT P_ID FROM PAYMENT WHERE B_ID=%s", (bno,))
                    pay_id=cur10.fetchall()
                    cur10.close()

                    payment_cursor.execute('SELECT calc_tot(%s)',(pay_id[0][0],))
                    tot_price=payment_cursor.fetchone()
                    total_payment=str(tot_price[0])                                          
            
                    messagebox.showinfo(title="Payment details for "+bno, message='Tip: '+str(tip)+"\n"+"Method: "+method+"\n"+"Extra: "+str(extra)+"\n"\
                                        +"Total payment: "+total_payment)

                    smain()

                p_frame=tk.Frame(sroot,height=700,width=1360)
                p_frame.place(x=171,y=121)
                bid_label=tk.Label(p_frame,text="Booking ID:",font=('Arial',18))
                bid_label.grid(row=0,column=0,padx=(180,5),pady=(170,30))
                bid_text=tk.Text(p_frame,font=('Arial',16),height=1,width=25)
                bid_text.grid(row=0,column=1,padx=(10,20),pady=(170,30))

                m_label=tk.Label(p_frame,text="Method:",font=('Arial',18))
                m_label.grid(row=1,column=0,padx=(180,5),pady=(100,30))
                meth=StringVar()
                method_drop=ttk.Combobox(p_frame,textvariable=meth,width=20,font=('Arial',18))
                method_drop.grid(row=1,column=1,padx=(10,20),pady=(100,30))
                method_drop['values']=['Cash','Card','Cheque']
            
                tip_label=tk.Label(p_frame,text="Tip:",font=('Arial',18))
                tip_label.grid(row=0,column=3,padx=(5,5),pady=(170,30))
                tip_text=tk.Text(p_frame,font=('Arial',16),height=1,width=25)
                tip_text.grid(row=0,column=4,padx=(10,100),pady=(170,30))     

                extra_label=tk.Label(p_frame,text="Extra Charges",font=('Arial',18))
                extra_label.grid(row=1,column=3,padx=(5,5),pady=(100,30))
                extra_text=tk.Text(p_frame,font=('Arial',16),height=1,width=25)
                extra_text.grid(row=1,column=4,padx=(10,100),pady=(100,30))   

                save_but=tk.Button(p_frame,text="Save",font=('Arial',16),width=10,command=save)
                save_but.grid(row=2,column=2,padx=(20,10),pady=(50,200))    
            def smain(): #main staff page after logging in 
                    s_main=tk.Frame(sroot,bg='#FAFAEB',height=700,width=1360)
                    s_main.place(x=171,y=121)
                    g_view=tk.Button(s_main,text="GUEST INFO",command=guest_view,font=('Arial',26))
                    g_view.pack(padx=550,pady=(75,20),side="top")
                    g_update=tk.Button(s_main,text= "UPDATE GUEST",command=update_guest,font=('Arial',26))
                    g_update.pack(padx=550,pady=20,side="top")
                    p_update=tk.Button(s_main,text="PAYMENT",command=payment,font=('Arial',26))
                    p_update.pack(padx=550,pady=20,side="top")
                    b_view=tk.Button(s_main,text="BOOKING INFO",command=checkbook,font=('Arial',26))
                    b_view.pack(padx=550,pady=20,side="top")
                    s_leave=tk.Button(s_main, text="LOGOUT", command=lambda: [sroot.destroy(), staff_login()], font=("Arial", 26))
                    s_leave.pack(padx=550, pady=(20,75), side="top")
          
            def staff_login_check(): 
                login_cursor=mydb.cursor()
                login_cursor.execute('SELECT USERNAME, PASSWORD FROM login') 
                login_values=login_cursor.fetchall() ##fetching all login details from the table
                login_cursor.close()
                username_password=[]
                for i in login_values:
                    username_password.append([i[0],i[1]])
                
                entered_login_details=[username_entry.get(),password_entry.get()]
                if entered_login_details in username_password:
                    smain()
                else:
                    messagebox.showerror(title="Wrong Password", message="Try Again")
                    sroot.withdraw()
                    staff_login()

            staff_login_check()


        login_label = tk.Label(frame, text="STAFF LOGIN", bg='#FFFFEF', fg="black", font=("Arial", 36))
        login_label.grid(row=0,column=0,padx=550,pady=(120,20),sticky='w')
        username_label = tk.Label(frame, text="USERNAME", bg='#FFFFEF', fg="black", font=("Arial", 16))
        username_label.grid(row=1,column=0,padx=550,pady=(20,5),sticky='w')
        username_entry = tk.Entry(frame, font=("Arial", 16),width=25)
        username_entry.grid(row=2,column=0,padx=550,pady=(0,20),sticky='w')
        password_label = tk.Label(frame, text="PASSWORD", bg='#FFFFEF', fg="black", font=("Arial", 16))
        password_entry = tk.Entry(frame, show="*", font=("Arial", 16),width=25)
        password_label.grid(row=3,column=0,padx=550,pady=(20,5),sticky='w')
        password_entry.grid(row=4,column=0,pady=(0,20),padx=550,sticky='w')
        login_button = tk.Button(frame, text="SUBMIT", bg="#CDCDC0", fg="black", font=("Arial", 16), command=login)
        login_button.grid(row=5,column=0,pady=(30,180),padx=550)
        
    def customer():
        def guestcreate():# function for guest create
            g_frame=tk.Frame(croot,bg='#FAFAEB',height=700,width=1360)
            g_frame.place(x=171,y=121)
            fname_label=tk.Label(g_frame,text="First Name:",font=('Arial',18))
            fname_label.grid(row=0,column=0,padx=(180,5),pady=(70,30))
            fname_text=tk.Text(g_frame,font=('Arial',16),height=1,width=25)
            fname_text.grid(row=0,column=1,padx=(10,20),pady=(70,30))
            
            lname_label=tk.Label(g_frame,text="Last Name:",font=('Arial',18))
            lname_label.grid(row=1,column=0,padx=(180,5),pady=(70,30))
            lname_text=tk.Text(g_frame,font=('Arial',16),height=1,width=25)
            lname_text.grid(row=1,column=1,padx=(10,20),pady=(70,30))
            
            age_label=tk.Label(g_frame,text="Age:",font=('Arial',18))   
            age_label.grid(row=2,column=0,padx=(250,5),pady=(70,30))
            age_text=tk.Text(g_frame,font=('Arial',16),height=1,width=25)
            age_text.grid(row=2,column=1,padx=(10,20),pady=(70,30))
            
            gender_label=tk.Label(g_frame,text="Gender:",font=('Arial',18))
            gender_label.grid(row=3,column=0,padx=(210,5),pady=(70,30))
            gen=StringVar()
            gender_drop=ttk.Combobox(g_frame,textvariable=gen,width=10,font=('Arial',16))
            gender_drop.grid(row=3,column=1,padx=(10,20),pady=(70,30),ipadx=70)
            gender_drop['values']=['Male','Female']
            
            email_label=tk.Label(g_frame,text="Email:",font=('Arial',18))
            email_label.grid(row=4,column=0,padx=(230,5),pady=(70,30))
            email_text=tk.Text(g_frame,font=('Arial',16),height=1,width=25)
            email_text.grid(row=4,column=1,padx=(10,20),pady=(70,30))
            phone_label=tk.Label(g_frame,text="Phone:",font=('Arial',18))
            phone_label.grid(row=0,column=3,padx=(220,5),pady=(70,30))
            phone_text=tk.Text(g_frame,font=('Arial',16),height=1,width=30)
            phone_text.grid(row=0,column=4,padx=(10,120),pady=(70,30))
            
            address_label=tk.Label(g_frame,text="Address:",font=('Arial',18))
            address_label.grid(row=1,column=3,padx=(200,5),pady=(70,30))
            address_text=tk.Text(g_frame,font=('Arial',16),height=1,width=30)
            address_text.grid(row=1,column=4,padx=(10,120),pady=(70,30))

            passport_label=tk.Label(g_frame,text="Passport No:",font=('Arial',18))
            passport_label.grid(row=2,column=3,padx=(160,5),pady=(70,30))
            passport_text=tk.Text(g_frame,font=('Arial',16),height=1,width=30)
            passport_text.grid(row=2,column=4,padx=(10,120),pady=(70,30))

            country_label=tk.Label(g_frame,text="Nationality:",font=('Arial',18))
            country_label.grid(row=3,column=3,padx=(180,10),pady=(70,30))
            country=StringVar()
            country_drop=ttk.Combobox(g_frame,textvariable=country,width=25,font=('Arial',16))
            country_drop.grid(row=3,column=4,padx=(10,120),pady=(70,30))
            country_drop['values']=['Afghanistan', 'Aland Islands', 'Albania', 'Algeria', 'American Samoa', 'Andorra', 'Angola', 'Anguilla', 'Antarctica', 'Antigua and Barbuda', 'Argentina', 'Armenia', 'Aruba', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin', 'Bermuda', 'Bhutan', 'Bolivia, Plurinational State of', 'Bonaire, Sint Eustatius and Saba', 'Bosnia and Herzegovina', 'Botswana', 'Bouvet Island', 'Brazil', 'British Indian Ocean Territory', 'Brunei Darussalam', 'Bulgaria', 'Burkina Faso', 'Burundi', 'Cambodia', 'Cameroon', 'Canada', 'Cape Verde', 'Cayman Islands', 'Central African Republic', 'Chad', 'Chile', 'China', 'Christmas Island', 'Cocos (Keeling) Islands', 'Colombia', 'Comoros', 'Congo', 'Congo, The Democratic Republic of the', 'Cook Islands', 'Costa Rica', "Côte d'Ivoire", 'Croatia', 'Cuba', 'Curaçao', 'Cyprus', 'Czech Republic', 'Denmark', 'Djibouti', 'Dominica', 'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Estonia', 'Ethiopia', 'Falkland Islands (Malvinas)', 'Faroe Islands', 'Fiji', 'Finland', 'France', 'French Guiana', 'French Polynesia', 'French Southern Territories', 'Gabon', 'Gambia', 'Georgia', 'Germany', 'Ghana', 'Gibraltar', 'Greece', 'Greenland', 'Grenada', 'Guadeloupe', 'Guam', 'Guatemala', 'Guernsey', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti', 'Heard Island and McDonald Islands', 'Holy See (Vatican City State)', 'Honduras', 'Hong Kong', 'Hungary', 'Iceland', 'India', 'Indonesia', 'Iran, Islamic Republic of', 'Iraq', 'Ireland', 'Isle of Man', 'Israel', 'Italy', 'Jamaica', 'Japan', 'Jersey', 'Jordan', 'Kazakhstan', 'Kenya', 'Kiribati', "Korea, Democratic People's Republic of", 'Korea, Republic of', 'Kuwait', 'Kyrgyzstan', "Lao People's Democratic Republic", 'Latvia', 'Lebanon', 'Lesotho', 'Liberia', 'Libya', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Macao', 'Macedonia, Republic of', 'Madagascar', 'Malawi', 'Malaysia', 'Maldives', 'Mali', 'Malta', 'Marshall Islands', 'Martinique', 'Mauritania', 'Mauritius', 'Mayotte', 'Mexico', 'Micronesia, Federated States of', 'Moldova, Republic of', 'Monaco', 'Mongolia', 'Montenegro', 'Montserrat', 'Morocco', 'Mozambique', 'Myanmar', 'Namibia', 'Nauru', 'Nepal', 'Netherlands', 'New Caledonia', 'New Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'Niue', 'Norfolk Island', 'Northern Mariana Islands', 'Norway', 'Oman', 'Pakistan', 'Palau', 'Palestinian Territory, Occupied', 'Panama', 'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines', 'Pitcairn', 'Poland', 'Portugal', 'Puerto Rico', 'Qatar', 'Réunion', 'Romania', 'Russian Federation', 'Rwanda', 'Saint Barthélemy', 'Saint Helena, Ascension and Tristan da Cunha', 'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Martin (French part)', 'Saint Pierre and Miquelon', 'Saint Vincent and the Grenadines', 'Samoa', 'San Marino', 'Sao Tome and Principe', 'Saudi Arabia', 'Senegal', 'Serbia', 'Seychelles', 'Sierra Leone', 'Singapore', 'Sint Maarten (Dutch part)', 'Slovakia', 'Slovenia', 'Solomon Islands', 'Somalia', 'South Africa', 'South Georgia and the South Sandwich Islands', 'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'South Sudan', 'Svalbard and Jan Mayen', 'Swaziland', 'Sweden', 'Switzerland', 'Syrian Arab Republic', 'Taiwan, Province of China', 'Tajikistan', 'Tanzania, United Republic of', 'Thailand', 'Timor-Leste', 'Togo', 'Tokelau', 'Tonga', 'Trinidad and Tobago', 'Tunisia', 'Turkey', 'Turkmenistan', 'Turks and Caicos Islands', 'Tuvalu', 'Uganda', 'Ukraine', 'United Arab Emirates', 'United Kingdom', 'United States', 'United States Minor Outlying Islands', 'Uruguay', 'Uzbekistan', 'Vanuatu', 'Venezuela, Bolivarian Republic of', 'Viet Nam', 'Virgin Islands, British', 'Virgin Islands, U.S.', 'Wallis and Futuna', 'Yemen', 'Zambia', 'Zimbabwe']
            def submit(): #function for button of submit in guest create
                
                def creategid():     #program to randomly generate in form of G101
                    cur1=mydb.cursor()
                    cur1.execute("SELECT MAX(G_ID) FROM GUEST")
                    newest=cur1.fetchone()
                    num=""
                    for s in newest:
                        for r in s:
                            if r.isdigit():
                                num+=r
                    next=int(num)+1
                    g_id="G"+str(next)
                    cur1.close()
                    return g_id

                g_id=creategid() 
                fname=fname_text.get(1.0,"end-1c")
                lname=lname_text.get(1.0,"end-1c")
                gender=gen.get()
                age=age_text.get(1.0,"end-1c")
                address=address_text.get(1.0,"end-1c")
                email=email_text.get(1.0,"end-1c")
                phone=phone_text.get(1.0,"end-1c")
                passno=passport_text.get(1.0,"end-1c")
                nationality=country.get()

                cur2=mydb.cursor()     #adding data into table 
                q1="INSERT INTO GUEST VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                cur2.execute(q1, (g_id, fname, lname, age, gender, nationality, phone, email, address, passno))
                mydb.commit()
                cur2.close()

                croot.withdraw()
                customer()
                temp=g_id   #only for display
                msg="Your Guest Id is:"
                messagebox.showinfo(title="Guest ID",message=msg+temp) 

            Submit_button=tk.Button(g_frame,text="Save",command=submit,font=('Arial',18))
            Submit_button.grid(row=4,column=3,padx=(180,10),pady=(70,30))
            

            def memberadd(): # button for switching to add memebers page
                def save(): # button for saving member data
                    gid=gid_text.get(1.0,"end-1c")
                    name=name_text.get(1.0,"end-1c")
                    passnum=pass_text.get(1.0,"end-1c")

                    cur2=mydb.cursor()
                    q2="INSERT INTO MEMBERS VALUES(%s, %s, %s)"
                    cur2.execute(q2, (gid, name, passnum))
                    mydb.commit()
                    cur2.close()
                    croot.withdraw()
                    customer()
                    messagebox.showinfo(title="Members",message="Member Added")
                    
                m_frame=tk.Frame(croot,bg='#FAFAEB',height=700,width=1360)
                m_frame.place(x=171,y=121)
                gid_label=tk.Label(m_frame,text="GID:",font=('Arial',18))
                gid_label.grid(row=0,column=2,padx=(550,0),pady=(150,50))
                gid_text=tk.Text(m_frame,font=('Arial',16),height=1,width=25)
                gid_text.grid(row=0,column=3,padx=(2,500),pady=(150,50))
                name_label=tk.Label(m_frame,text="Name:",font=('Arial',18))
                name_label.grid(row=1,column=2,padx=(520,0),pady=(50,50))
                name_text=tk.Text(m_frame,font=('Arial',16),height=1,width=25)
                name_text.grid(row=1,column=3,padx=(2,500),pady=(50,50))
                pass_label=tk.Label(m_frame,text="Passport No:",font=('Arial',18))
                pass_label.grid(row=2,column=2,padx=(450,0),pady=(50,50))
                pass_text=tk.Text(m_frame,font=('Arial',16),height=1,width=25)
                pass_text.grid(row=2,column=3,padx=(2,500),pady=(50,50))
                save_button=tk.Button(m_frame,text="Save",font=('Arial',18),command=save)
                save_button.grid(row=3,column=3,padx=(0,500),pady=(50,330))
            members_button=tk.Button(g_frame,text="Add Members",font=('Arial',18),command=memberadd)
            members_button.grid(row=4,column=4,padx=(10,20),pady=(70,30))

        def newbook(): # function to open page for new booking 
            b_frame=tk.Frame(croot,bg='#FAFAEB',height=700,width=1360)
            b_frame.place(x=171,y=121)

            def bsubmit(): #function to save new booking 
                                    
                gid=guest_text.get(1.0,"end-1c")
                check_in=checkin_cal.get_date()
                check_out=checkout_cal.get_date()
                type=siz.get()
                description=vi.get()

                def checkroom():  #find a room based on type and description
                    cur3=mydb.cursor(buffered=True)
                    q3="SELECT R_NO FROM ROOM WHERE TYPE=%s && DESCRIPTION=%s && STATUS='VACANT'"
                    cur3.execute(q3, (type, description))
                    room=cur3.fetchone()     
                    return room[0]
                
                def createbid():        #create new b_id
                    cur4=mydb.cursor(buffered=True)
                    cur4.execute("SELECT MAX(B_ID) FROM BOOKING")
                    newestB=cur4.fetchone()
                    numB=""
                    for s in newestB:
                        for r in s:
                            if r.isdigit():
                                numB+=r
                    next=int(numB)+1
                    b_id="B"+str(next)
                    cur4.close()
                    return b_id
                    
                bid=createbid()
                rno=checkroom()

                now = datetime.now()
                dt_string = now.strftime("%y-%m-%d %H:%M:%S")

                cur5=mydb.cursor()
                q5="INSERT INTO BOOKING (B_ID,G_ID,R_NO,B_DATE,CHECK_IN,CHECK_OUT) VALUES (%s, %s, %s, %s, %s, %s)"
                cur5.execute(q5, (bid, gid, rno, dt_string, check_in, check_out))
                mydb.commit()

                croot.withdraw()
                customer()
                messagebox.showinfo(title="Booking",message="Booking ID: "+bid)
                  
            guest_label=tk.Label(b_frame,text="Guest ID:",font=('Arial',18))
            guest_label.grid(row=0,column=0,padx=(200,5),pady=(70,30))
            guest_text=tk.Text(b_frame,font=('Arial',16),height=1,width=25)
            guest_text.grid(row=0,column=1,padx=(5,20),pady=(70,30))
            checkin_label=tk.Label(b_frame,text="Check-In Date:",font=('Arial',18))
            checkin_label.grid(row=1,column=0,padx=(180,5),pady=(50,30))
            checkin_cal=Calendar(b_frame,selectmode="day",year=2023,month=1,date=1,date_pattern="yyyy-mm-dd")
            checkin_cal.grid(row=1,column=1,padx=(5,20),pady=(50,30))
            checkout_label=tk.Label(b_frame,text="Check-Out Date:",font=('Arial',18))
            checkout_label.grid(row=2,column=0,padx=(180,5),pady=(50,30))
            checkout_cal=Calendar(b_frame,selectmode="day",year=2023,month=1,date=1,date_pattern="yyyy-mm-dd")
            checkout_cal.grid(row=2,column=1,padx=(5,20),pady=(50,30))
            
            size=tk.Label(b_frame,text="Room Size",font=('Arial',18))
            size.grid(row=0,column=3,padx=(120,0),pady=(70,30))
            siz=StringVar()
            size_drop=ttk.Combobox(b_frame,textvariable=siz,width=25,font=('Arial',16))
            size_drop.grid(row=0,column=4,padx=(10,100),pady=(70,30))
            size_drop['values']=['STUDIO','STANDARD','DELUXE','SUITE']

            view=tk.Label(b_frame,text="Room View:",font=('Arial',18))
            view.grid(row=1,column=3,padx=(120,0),pady=(70,30))
            vi=StringVar()
            view_drop=ttk.Combobox(b_frame,textvariable=vi,width=25,font=('Arial',16))
            view_drop.grid(row=1,column=4,padx=(10,100),pady=(70,30))
            view_drop['values']=['Pool Side', 'Theme Park', 'Beach Side', 'Track View', 'City View']
            submit_button=tk.Button(b_frame,text="Book",font=('Arial',18),command=bsubmit)
            submit_button.grid(row=2,column=4,padx=(0,0),pady=(70,30),ipadx=40)

        def checkbook(): # function to open check booking 
            c_frame=tk.Frame(croot,height=700,width=1360)
            c_frame.place(x=171,y=121)
            label1=tk.Label(c_frame,text="BOOKING NUMBER:",font=('Arial',18))
            label1.pack(side='left',padx=(350,20),pady=(300,325))
            bookingno=tk.Text(c_frame,font=('Arial',18),height=1,width=25)
            bookingno.pack(side='left',padx=25,pady=(300,325))
            def takebno(): #check button function                                
                bno=bookingno.get(1.0,"end-1c")                                     

                cur6=mydb.cursor()
                q6="SELECT * FROM BOOKING WHERE B_ID=%s"       
                cur6.execute(q6, (bno,))
                booking1=cur6.fetchall()
                bgid=booking1[0][1]                                                
                brno=booking1[0][2]                                                  
                bdate=str(booking1[0][3])
                bcheckin=str(booking1[0][4])                                            
                bcheckout=str(booking1[0][5])                                       
                bdur=str(booking1[0][6])
                cur6.close()
                
                cur7=mydb.cursor()
                q7="SELECT TYPE, DESCRIPTION, PRICE FROM ROOM WHERE R_NO=%s"
                cur7.execute(q7, (brno,))
                booking2=cur7.fetchall()
                btype=booking2[0][0]                                                 
                bdesc=booking2[0][1]                                                 
                bprice=str(booking2[0][2])                                           
                cur7.close()


                messagebox.showinfo(title="Booking details for "+bno, message='Guest id: '+bgid+"\n"+"Room number: "+brno \
                                        +"\n"+"Booking Date: "+bdate+"\n"+"Check-in Date: "+bcheckin+"\n"+"Check-out Date: "+bcheckout+"\n"\
                                        +"Duration of Stay: "+bdur+"\n"+"\n"+"Room Type: "+btype+"\n"+"Room Description: "+bdesc+"\n"+"Price: "+bprice)

            
            check=tk.Button(c_frame,text="Check",font=('Arial',16),width=10,command=takebno)
            check.pack(padx=20,pady=(300,325),side='left')

            exit=tk.Button(c_frame,text="Exit",font=('Arial',16),width=10, command=lambda : [croot.destroy(), customer()])
            exit.pack(padx=(20,1000),pady=(300,325), side='left')

        #Start of customer main page
        croot=tk.Toplevel(root)
        croot.title("Customer")
        croot.geometry("1920x1080") #whole window
        root.withdraw()
        vertical=tk.Frame(croot,bg='#A2A2CD',height=1080,width=170) 
        vertical.place(x=0,y=0)
        heading=tk.Frame(croot,bg='#A2A2CD')
        heading.pack()
        main_logo=ImageTk.PhotoImage((Image.open("Images\logo.png")).resize((100,100)))
        picture=tk.Label(heading,image=main_logo,bg='#A2A2CD')
        picture.photo=main_logo
        picture.pack(side="left",ipadx=30)
        title=tk.Label(heading,text="Shangri-Nah Hotel",font=('Helvetica',36,'bold'),bg='#A2A2CD')
        title.pack(padx=475,pady=30)

        c_main=tk.Frame(croot,bg='#FAFAEB',height=700,width=1360)
        c_main.place(x=171,y=121)
        
        g_create=tk.Button(c_main,text="NEW GUEST",command=guestcreate,font=('Arial',26))
        g_create.pack(padx=550,pady=75,side="top")
        b_create=tk.Button(c_main,text="NEW BOOKING",command=newbook,font=('Arial',26))
        b_create.pack(padx=550,pady=75,side="top")
        b_check=tk.Button(c_main,text="CHECK BOOKING",command=checkbook,font=('Arial',26))
        b_check.pack(padx=550,pady=80,side="top")

    #Start of main page code
    root=tk.Tk()
    root.title("Shangri-Nah")
    root.geometry("1000x600")
    staff_image=Image.open("Images\staff.png")
    staff_image1=staff_image.resize((250,250))
    Staff_logo=ImageTk.PhotoImage(staff_image1)
    Staff_button=tk.Button(root,image=Staff_logo,command=staff_login,borderwidth=0,height=250,width=250)
    Staff_button.grid(row=0,column=0,padx=175,pady=150)

    cust_image=Image.open("Images\Cust.png")
    cust_image1=cust_image.resize((250,250))
    Cust_logo=ImageTk.PhotoImage(cust_image1)
    Cust_button=tk.Button(root,image=Cust_logo,command=customer,borderwidth=0,height=250,width=250)
    Cust_button.grid(row=0,column=1)
    root.mainloop()
main()