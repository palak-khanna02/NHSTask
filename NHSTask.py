#!/usr/bin/env python
# coding: utf-8





# Import Libraries

from math import *
import time
from datetime import date
import pandas as pd 
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


st.title('NHS Blood Donation Task')



# Import Data

datapath = '/Users/admin/Downloads/'

reg = pd.read_csv('./registrations.csv')
slots =pd.read_csv('./slots.csv')


# Format Date and Time columns for easier manipulation


reg['REGDAT'] = pd.to_datetime(reg['REGDAT'], format='%Y%m%d')
reg['DOB'] = pd.to_datetime(reg['DOB'], format='%Y%m%d')

slots['SESDATE'] = pd.to_datetime(slots['SESDATE'], format='%Y%m%d')
slots['APPTIME'] = pd.to_datetime(slots['APPTIME'], format='%H%M').apply(lambda x: x.time())

st.header('Donor and Slots Data')
if st.toggle(label='Data'):
    st.dataframe(reg)
    st.dataframe(slots)

#############################

st.divider()

############################

st.header('Data Quality Checks')

st.subheader('Checking Missing Values and Duplicates')

if st.toggle(label='Initial Checks'):
    st.write(reg.info())
    dup= reg.duplicated(subset=['ID']).sum()
    st.write(f'No. of Duplicate Entries in registrations: {dup}')
    st.write(f'No. of Duplicate Entries in slots: {slots.duplicated().sum()}')
    st.write(reg.isnull().sum())
    st.write(slots.isnull().sum())
    reg.describe()


############################

st.divider()

############################


st.header('Trend of Registrations')


# Resample the data to show No. of registrations per week 

df = reg.set_index('REGDAT').resample('W').size()


# Plot Trend of Registrations Over Time 

chart= px.line(df,  x=df.index, y=[0], title='Registrations over Time (Weekly)', markers=True)
st.plotly_chart(chart)

text1= '''-  Sharp increase in mid October, likely due to NHS Black History Month Campaigns   
  -  Low Registration throughout Winter months.
- Slight increase throughout January, owing to New Year campaigns 
- Increase going into March'''




st.markdown(text1)

############################

st.divider()

############################


st.header('Demographic Analysis')
# Demographic Anlaysis


st.subheader('Analysing Donations by Blood Groups')


# Plot Blood Group Distribution Registered People
reg.fillna({'BLOOD_GROUP': 'Unknown'}, inplace=True)

blood= px.pie(reg, names='BLOOD_GROUP', title='Percentage Registered People by Blood Group')
#blood.update_traces(histfunc="count")
st.plotly_chart(blood)

text2= '''- Maximum percentage are unknown, expect a majority of them to made no donations.
 -  Registrations from common Blood Groups :red[O+, A+, B+] 
  -  Discrepancies in data collection, few entries with incomplete blood types'''


st.markdown(text2)

# Plot Donation behavior by blood group 

blood_don= px.pie(reg, values='NUM_DONATIONS', names='BLOOD_GROUP', title='Percentage of Donations by Blood Group')
st.plotly_chart(blood_don)

text3 =  '''-  Donation behaviour follows a similar pattern with :red[O+, A+, B+] forming maximum donations
  - Blood group unknown implies no donations except a few entries with incomplete blood types even after donation '''
st.markdown(text3)

# Calculate the number of donors for each blood group
donors_per_blood_group = reg.groupby('BLOOD_GROUP')['ID'].nunique().reset_index()
donors_per_blood_group.columns = ['BLOOD_GROUP', 'NUM_DONORS']

# Calculate the total number of donations for each blood group
donations_per_blood_group = reg.groupby('BLOOD_GROUP')['NUM_DONATIONS'].sum().reset_index()
donations_per_blood_group.columns = ['BLOOD_GROUP', 'TOTAL_DONATIONS']

# Merge the dataframes
blood_group_data = pd.merge(donors_per_blood_group, donations_per_blood_group, on='BLOOD_GROUP')

# Data for the plot
blood_groups = blood_group_data['BLOOD_GROUP']
num_donors = blood_group_data['NUM_DONORS']
total_donations = blood_group_data['TOTAL_DONATIONS']

# Create the figure
blood_bar = go.Figure(data=[
    go.Bar(name='Number of Donors', x=blood_groups, y=num_donors, marker_color='indianred'),
    go.Bar(name='Total Donations', x=blood_groups, y=total_donations, marker_color='lightblue')
])

# Update the layout
blood_bar.update_layout(
    title='Number of Donors and Total Donations by Blood Group',
    xaxis_title='Blood Group',
    yaxis_title='Count',
    barmode='group',
    legend_title='Metric'
)

st.plotly_chart(blood_bar)
# Plot Donation Patterns by Blood Group Box plot 


blood_box = px.box(reg, x="BLOOD_GROUP", y="NUM_DONATIONS", title='Donation Patterns by Blood Group')
st.plotly_chart(blood_box)

text4 =  '''-  Most people with rare blood like :green[O- , B-, AB+ , AB-] have made a minimum of 1 donation. 
- Rarer blood types :green[AB-, A-, B-] show fewer high-frequency donors.
- Some donors with common blood groups :red[O+, B+] have made no donations. 
- Very few donors have donate more than 6 times. 
- More than 50 percent of donors in known blood group categories have donated more than 1 time '''

st.markdown(text4)

st.caption('Areas of Focus and Recommendations:')
text11='''
- Imporvement in data collection, blood groups are unknown even after donation.  
- Increase donations from rare blood types, providing special incentives, recognition encouraging frequent donation.  
- Partnering with Universities and workplaces to spread awareness about rare blood types    
- Mobile donation/information camps near Emergency, encourage people accompanying patients to register due to long queues and wait times 
  - Offering refreshments, options to donate or book a future date
  - Free blood type tests post registration, incentivise rare types providing vouchers. '''

st.markdown(text11)
############################

st.divider()

############################


st.header('Donation behaviour by Ethnic Groups')





# Create the DID_NOT_BOOK column
# If NUM_DONATIONS, NUM_DNAS and NUM_REJECTIONS are all 0 , means the person registered but made no appointments

reg['DID_NOT_BOOK'] = ((reg['NUM_DONATIONS'] == 0) & (reg['NUM_DNAS'] == 0) & (reg['NUM_REJECTIONS'] == 0)).astype(int)




# Segregate Ethnic Groups based on Ethnicity

reg["ETHNIC_GROUP"] = ""
for row, ethnicity in enumerate(reg['ETHNICITY']):
    if "Mixed" in ethnicity:
        reg['ETHNIC_GROUP'][row]= 'Mixed'
    elif 'Black' in ethnicity:
        reg['ETHNIC_GROUP'][row]= 'Black'
    elif 'Asian' in ethnicity or 'Chinese' in ethnicity:
        reg['ETHNIC_GROUP'][row]= 'Asian'
    elif 'Gypsy' in ethnicity or 'White' in ethnicity or 'Brit' in ethnicity:
        reg['ETHNIC_GROUP'][row]= 'White'
    else:
        reg['ETHNIC_GROUP'][row]= 'Other'
    
    


# Calculate the number of donors for each ethnic group
donors_per_ethnic_group = reg.groupby('ETHNIC_GROUP')['ID'].nunique().reset_index()
donors_per_ethnic_group.columns = ['ETHNIC_GROUP', 'NUM_DONORS']

# Calculate the total number of donations for each ethnic group
donations_per_ethnic_group = reg.groupby('ETHNIC_GROUP')['NUM_DONATIONS'].sum().reset_index()
donations_per_ethnic_group.columns = ['ETHNIC_GROUP', 'TOTAL_DONATIONS']

# Merge the dataframes
ethnic_group_data = pd.merge(donors_per_ethnic_group, donations_per_ethnic_group, on='ETHNIC_GROUP')

# Calculate the percentage
ethnic_group_data['PERCENTAGE_DONORS'] = (ethnic_group_data['NUM_DONORS'] /donors_per_ethnic_group['NUM_DONORS'].sum()) * 100
ethnic_group_data['PERCENTAGE_DONATIONS'] = (ethnic_group_data['TOTAL_DONATIONS'] / donations_per_ethnic_group['TOTAL_DONATIONS'].sum()) * 100

# Data for the plot
ethnic_groups = ethnic_group_data['ETHNIC_GROUP']
percentage_donors = ethnic_group_data['PERCENTAGE_DONORS']
percentage_donations = ethnic_group_data['PERCENTAGE_DONATIONS']

# Create the figure
eth_bar = go.Figure(data=[
    go.Bar(name='Percentage of Donors', x=ethnic_groups, y=percentage_donors, marker_color='indianred'),
    go.Bar(name='Percentage of Donations', x=ethnic_groups, y=percentage_donations, marker_color='lightblue')
])

# Update the layout
eth_bar.update_layout(
    title='Percentage of Donors and Donations by Ethnic Group',
    xaxis_title='Ethnic Group',
    yaxis_title='Percentage',
    barmode='group',
    legend_title='Metric'
)
st.plotly_chart(eth_bar)
# Plot Registered Ethnic Groups 

text5='''- Maximum registrations and donation from White Ethnic Groups.  
- Other Ethnic groups show more proportion of registration and less contribution to donantions.'''

st.markdown(text5)

# Create a pivot table to analyse Percentage of Donations, Rejections, Missed Appointments, No Bookings by Ethnic Groups
table = pd.pivot_table(reg, values=['NUM_DONATIONS', 'NUM_DNAS', 'NUM_REJECTIONS', 'DID_NOT_BOOK'], columns=['ETHNIC_GROUP'], aggfunc="sum").apply(lambda x: x*100/sum(x))


# Plot pivot table 
stack = px.bar(table.T ,text_auto='.3s',title="Percentage of Donations, Rejections, Missed Appointments, No Bookings by Ethnic Groups")
stack.update_layout(xaxis_title='Ethnic Group',yaxis_title='Percentage')
st.plotly_chart(stack)

text6= '''- Most Groups register and book appointments, percentage of no bookings is low across all groups.  
- No shows are mainly from Asian and Black ethnic groups followed by Other and Mixed. 
- No shows are high across all groups '''

st.markdown(text6)
st.caption('Areas of Focus and Recommendations:')
text12='''
-  Targeted outreach programs for underrepresented ethnic groups partnering with community organizations, religious or cultural centers.  
-  Investigating reasons for higher DNA rates among Asian, Black, and Mixed groups, implementing reminder systems
-  Mobile Donation drives in areas of high concentrations of these groups. '''

st.markdown(text12)

#####################

st.divider()

#####################


st.header('Donation behaviour by Age')

reg['AGE']= (date.today().year - reg['DOB'].dt.year)
reg['AGE_GROUP'] = pd.cut(reg['AGE'], bins=[0, 18, 30, 45, 60, 100], labels=['<18', '18-30', '30-45', '45-60', '60+'])
reg['DONOR_TYPE'] = np.where(reg['NUM_DONATIONS'] > 1, 'Repeat Donor',np.where(reg['NUM_DONATIONS'] == 1, 'One-Time Donor', 'Non-Donor'))




# Calculate the number of donors for each age group
donors_per_age_group = reg.groupby('AGE_GROUP')['ID'].nunique().reset_index()
donors_per_age_group.columns = ['AGE_GROUP', 'NUM_DONORS']

# Calculate the total number of donations for each age group
donations_per_age_group = reg.groupby('AGE_GROUP')['NUM_DONATIONS'].sum().reset_index()
donations_per_age_group.columns = ['AGE_GROUP', 'TOTAL_DONATIONS']

# Merge the dataframes
age_group_data = pd.merge(donors_per_age_group, donations_per_age_group, on='AGE_GROUP') 

# Calculate the percentage
age_group_data['PERCENTAGE_DONORS'] = (age_group_data['NUM_DONORS'] / donors_per_age_group['NUM_DONORS'].sum()) * 100
age_group_data['PERCENTAGE_DONATIONS'] = (age_group_data['TOTAL_DONATIONS'] / donations_per_age_group['TOTAL_DONATIONS'].sum()) * 100

# Data for the plot
age_groups = age_group_data['AGE_GROUP']
percentage_donors = age_group_data['PERCENTAGE_DONORS']
percentage_donations = age_group_data['PERCENTAGE_DONATIONS']

# Create the figure
age_bar = go.Figure(data=[
    go.Bar(name='Percentage of Donors', x=age_groups, y=percentage_donors, marker_color='indianred'),
    go.Bar(name='Percentage of Donations', x=age_groups, y=percentage_donations, marker_color='lightsalmon')
])

# Update the layout
age_bar.update_layout(
    title='Percentage of Donors and Donations by Age Group',
    xaxis_title='Age Group',
    yaxis_title='Percentage',
    barmode='group',
    legend_title='Metric'
)

st.plotly_chart(age_bar)

text7='''- Young Adults and Middle ages show highest engagement and donations.  
- Proportion of donations to donors is higher in Older Adults (45-60 and 60+) implying more repeat donors.  
- Teenagers show very low donation counts, as individuals under 18 are often not eligible to donate blood. '''

st.markdown(text7)

# Box plot for age group vs. number of donations

Age_box = px.box(reg, x="AGE_GROUP", y="NUM_DONATIONS", title='Donation Patterns by Age Group')
Age_box.update_xaxes(categoryorder='array', categoryarray= ['<18', '18-30', '30-45', '45-60', '60+'])
st.plotly_chart(Age_box)

text8='''- Median is 0 for Young Adults and Middle ages indicating at least 50 percent of donors did not donate.  
- More consistent and repeat donors from higher age groups.'''

st.markdown(text8)
table_age = pd.pivot_table(reg, values=['NUM_DONATIONS', 'NUM_DNAS', 'NUM_REJECTIONS', 'DID_NOT_BOOK'], columns=['AGE_GROUP'], aggfunc="sum").apply(lambda x: x*100/sum(x))

stack_age = px.bar(table_age.T ,text_auto='.3s',title="Percentage of Donations, Rejections, Missed Appointments, No Bookings by Age Groups")
stack.update_traces(text= [f'{val}\u00A3' for val in table_age.T])
st.plotly_chart(stack_age)

stack_age.update_layout(
    xaxis_title='Age Group',
    yaxis_title='Percentage',
    legend_title='Metric'
)

text9='''- Older Adults more successful in keeping appointments.  
- Rejection rates in 60+ groups lower than Teenagers and Adults. 
- Maximum no shows from Young Adults, common reasons could be university or work commitments.  '''
st.markdown(text9)
# Calculate percentages for age groups
age_group_counts = reg.groupby(['AGE_GROUP', 'DONOR_TYPE']).size().unstack().fillna(0)
age_group_percentages = age_group_counts.div(age_group_counts.sum(axis=1), axis=0) * 100


# Create the figure
agetype_bar = go.Figure(data=[
    go.Bar(name='Non-Donors', x=age_groups, y=age_group_counts['Non-Donor'], marker_color='indianred'),
    go.Bar(name='One-Time Donors', x=age_groups, y=age_group_counts['One-Time Donor'], marker_color='lightsalmon'),
    go.Bar(name='Repeat-Donors', x=age_groups, y=age_group_counts['Repeat Donor'], marker_color='lightblue'),
])

# Update the layout
agetype_bar.update_layout(
    title='Number of Non-Donors, One-time Donors and Repeat-Donors by Age Group',
    xaxis_title='Age Group',
    yaxis_title='Count',
    barmode='group',
    legend_title='Metric'
)

st.plotly_chart(agetype_bar)
text10='''
- Older Adults form a more reliable donor base, high percentage of repeat donors. 
- More than 50%  of registered Young Adults and Middle aged adults are Non-Donors.
- Even though Number of donors is greater in Young and Middle ages Older age groups are more likely to be repeat donors (plot below). '''

st.markdown(text10)

stack_type=px.bar(age_group_percentages, text_auto='.2s')
stack_type.update_layout(
    xaxis_title='Age Group',
    yaxis_title='Percentage',
    legend_title='Metric')

st.plotly_chart(stack_type)

st.caption('Areas of Focus and Recommendations:')
text13='''
 - Further investigation into why Non-donors or One-time donors do not donate. 
 - Conversion of young donors, partnering with Schools and Universities to spread awareness and importance of donation. 
 - Incentives to University students like discounts/freebies for donating  
 - Partnering with Corporate social resposibility departments in private organisations to hold donation drives within corporations. 
 - Offer pre-screening or free general health checkup to encourage doantion and decrease chances of rejection. '''

st.markdown(text13)

######################

st.divider()

######################

# Geographical distribution of donors (example for one center)

st.header('Geographical analysis')
''' We make the assumption here that a donor will book and visit the Donor Center closest to them. Moving forward with this assumption we can find the closest Donor center from each donor and alot them a Catchment area'''
distance_cols = ['DISTANCE_CORNWALL', 'DISTANCE_CANTERBURY', 'DISTANCE_PRESTON', 'DISTANCE_LEWISHAM', 'DISTANCE_DURHAM']


# Calculate minimum distance and corresponding catchment area
reg['MIN_DIST'] = reg[distance_cols].min(axis=1)
reg['CATCHMENT'] = reg[distance_cols].idxmin(axis=1).str.replace('DISTANCE_', '')



# Aggregating data
area_data = reg.groupby('CATCHMENT').agg(Total_Donors=pd.NamedAgg(column='ID', aggfunc='count'), Did_Not_Show=pd.NamedAgg(
    column='NUM_DNAS', aggfunc='sum'), Rejections=pd.NamedAgg(column='NUM_REJECTIONS', aggfunc='sum'),
    Successful_Donations=pd.NamedAgg(column='NUM_DONATIONS', aggfunc='sum')).reset_index()

# Calculate additional metrics
area_data['Appointments_Booked']= area_data.Did_Not_Show+ area_data.Rejections+area_data.Successful_Donations 
area_data['Appointments_Attended'] = area_data['Appointments_Booked'] - area_data['Did_Not_Show']
area_data['Successful_Donations'] = area_data['Appointments_Attended'] - area_data['Rejections']
area_data.set_index('CATCHMENT', inplace=True)

# Count available slots by donor center
slots_by_area = slots.groupby('DC')['SLOT'].sum()

# Merge area_data and slots_by_area 

area= area_data.merge(slots_by_area.rename('Available_Slots'), left_index=True, right_index=True)


# Create the figure
area_bar = go.Figure(data=[
    go.Bar(name='Appointments Booked', x=area.index, y=area['Appointments_Booked'], marker_color='indianred'),
    go.Bar(name='Appointments Attended', x=area.index, y=area['Appointments_Attended'], marker_color='lightsalmon'),
    go.Bar(name='Successful Donations', x=area.index, y=area['Successful_Donations'], marker_color='lightblue')
])

area_bar.update_layout(
    title='Blood Donation success by Area',
    xaxis_title='Area',
    yaxis_title='Value',
    legend_title='Metric'
)

st.plotly_chart(area_bar)
text15='''
- Lewisham significantly outperforms other areas in terms of appointments booked, attended, and successful donations  
- High No show rate in Lewisham could suggest other problems, possible over-scheduling, staff shortage
- Durham: Shows the second-highest performance, but with a significant gap
- Cornwall, Preston, and Canterbury: Have relatively similar, lower levels of performance  
- Gap between appointments attended and successful donations is smaller, suggesting that once people attend, they're likely to complete a successful donation.
'''
st.markdown(text15)

slots_bar = go.Figure(data=[
    go.Bar(name='Appointments Booked', x=area.index, y=area['Appointments_Booked'], marker_color='indianred'),
    go.Bar(name='Slot Availibility', x=area.index, y=area['Available_Slots'], marker_color='lightsalmon' )
])


st.plotly_chart(slots_bar)

text18='''
- Slot availibity in Lewisham and Durham is lower compared to previously booked appointments
- Resources can be diverted to high performing areas '''

st.markdown(text18)

st.caption('Areas of Focus and Recommendations:')
text14='''
- Review the booking processes in Lewisham to avoid overbooking.  
- Implement a waitlist system to fill cancelled appointments quickly.  
- Parnering with local employers in each areas to offer employees paid time off (few hours)for blood donation.  
- Offer incentives linke gift cards if donors bring friends or family for donation.    
 '''

st.markdown(text14)
###########################

st.divider()

###########################
st.header("Ro Type Donation Patterns")

# Filter for Ro type blood donors
ro_donors_df = reg[reg['IS_RO'] == 1]

# Calculate the percentage of registered Ro donors who made at least one donation
ro_donors_made_donation = ro_donors_df[ro_donors_df['NUM_DONATIONS'] > 0]
percentage_ro_donors_made_donation = (ro_donors_made_donation.shape[0] / ro_donors_df.shape[0]) * 100


st.metric(label="Percentage of registered Ro donors who made a donation:", value=f"{percentage_ro_donors_made_donation:.2f}%")

repeat_donor_counts_ro_actual = ro_donors_made_donation['DONOR_TYPE'].value_counts(normalize=True) * 100

Ro_donor_type=px.bar(repeat_donor_counts_ro_actual, text_auto='.4s')
st.plotly_chart(Ro_donor_type)

text16='''- Most Ro Donors who register have donated. 
- More than 50 percent are repeat donors 
- Contrary to previous age patterns, Most Ro donors are Young or Middle aged (plot below)'''
st.markdown(text16)
Ro_donor_age=px.bar(ro_donors_df['AGE_GROUP'].value_counts())
Ro_donor_age.update_xaxes(categoryorder='array', categoryarray= ['<18', '18-30', '30-45', '45-60', '60+'])
st.plotly_chart(Ro_donor_age)
st.caption('Areas of Focus and Recommendations and Further Work:')
text17='''
- More campaigns targeted at people with Black heritage and targeting a younger audience for social resposibility  
   - Eg. NHSBTs partnership with Marvel Studios' Black Panther: Wakanda Forever
- Campaigns in areas with high concentration of people of Black heritage, urging them to get tested and become donors
- Volunteer programs for young and frequent donors to conduct social outreach through volunteering at donations camps, spreading awareness through social media  
**Further Work:**  
- Research into reasons for No shows and not donating more than once
- Analysing post donoation surveys of donor experience
- Hypothesis testing 

'''
st.markdown(text17)
st.divider()
