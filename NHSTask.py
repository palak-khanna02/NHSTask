#!/usr/bin/env python
# coding: utf-8





# Import Libraries

from math import *
import time
from datetime import date
import pandas as pd 
import numpy as np
import streamlit as st



st.title('NHS Blood Donation Task')



# Import Data

datapath = '/Users/admin/Downloads/'

reg = pd.read_csv(datapath+'registrations.csv')
slots =pd.read_csv(datapath+'slots.csv')


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
- Increase going into :red[March]'''




st.markdown(text1)

############################

st.divider()

############################


st.header('Demographic Analysis')
# Demographic Anlaysis


st.subheader('Analysing Donations by Blood Groups')


# Plot Blood Group Distribution Registered People


blood= px.histogram(reg, x='BLOOD_GROUP', text_auto=True, title='Blood Group Distribution Registered People')
blood.update_traces(histfunc="count")
st.plotly_chart(blood)



# Plot Donation behavior by blood group 

blood_don= px.pie(reg, values='NUM_DONATIONS', names='BLOOD_GROUP', title='Donation behaviour by Blood Group')
st.plotly_chart(blood_don)

# Plot Donation Patterns by Blood Group Box plot 


blood_box = px.box(reg, x="BLOOD_GROUP", y="NUM_DONATIONS", title='Donation Patterns by Blood Group')
st.plotly_chart(blood_box)

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
    
    
# Plot Registered Ethnic Groups 

ETH_reg_gr = px.pie(reg, names='ETHNIC_GROUP', title='Registered Ethnic Groups')
st.plotly_chart(ETH_reg_gr)

# Plot Donation Behaviour of Ethnic Groups

ETH_don_gr = px.pie(reg, values='NUM_DONATIONS', names='ETHNIC_GROUP', title='Donation Behaviour of Ethnic Groups')
st.plotly_chart(ETH_don_gr)

# Create a pivot table to analyse Percentage of Donations, Rejections, Missed Appointments, No Bookings by Ethnic Groups
table = pd.pivot_table(reg, values=['NUM_DONATIONS', 'NUM_DNAS', 'NUM_REJECTIONS', 'DID_NOT_BOOK'], columns=['ETHNIC_GROUP'], aggfunc="sum").apply(lambda x: x*100/sum(x))
st.write(table)

# Plot pivot table 
stack = px.bar(table.T ,text_auto='.3s',title="Percentage of Donations, Rejections, Missed Appointments, No Bookings by Ethnic Groups")
stack.update_traces(text= [f'{val}\u00A3' for val in table.T])
st.plotly_chart(stack)


#####################

st.divider()

#####################


st.header('Donation behaviour by Age')

reg['AGE']= (date.today().year - reg['DOB'].dt.year)
reg['AGE_GROUP'] = pd.cut(reg['AGE'], bins=[0, 18, 30, 45, 60, 100], labels=['<18', '18-30', '30-45', '45-60', '60+'])
reg['DONOR_TYPE'] = np.where(reg['NUM_DONATIONS'] > 1, 'Repeat Donor',np.where(reg['NUM_DONATIONS'] == 1, 'One-Time Donor', 'Non-Donor'))



Age_reg = px.histogram(reg, x='AGE', title='Registered Donors by Age', text_auto=True)
st.plotly_chart(Age_reg)

Age_don=px.histogram(reg, x='AGE', y='NUM_DONATIONS',  text_auto=True, title='Donation behaviour by Ethnicity')
st.plotly_chart(Age_don)


Age_reg_gr = px.pie(reg, names='AGE_GROUP', title='Registered Age Groups')
st.plotly_chart(Age_reg_gr)


Age_gr_don = px.pie(reg, values='NUM_DONATIONS', names='AGE_GROUP', title='Donation Behaviour of Age Groups')
st.plotly_chart(Age_gr_don)

# Box plot for age group vs. number of donations

Age_box = px.box(reg, x="AGE_GROUP", y="NUM_DONATIONS", title='Donation Patterns by Age Group')
st.plotly_chart(Age_box)

table_age = pd.pivot_table(reg, values=['NUM_DONATIONS', 'NUM_DNAS', 'NUM_REJECTIONS', 'DID_NOT_BOOK'], columns=['AGE_GROUP'], aggfunc="sum").apply(lambda x: x*100/sum(x))
st.write(table_age)

stack_age = px.bar(table_age.T ,text_auto='.3s',title="Percentage of Donations, Rejections, Missed Appointments, No Bookings by Age Groups")
stack.update_traces(text= [f'{val}\u00A3' for val in table_age.T])
st.plotly_chart(stack_age)

# Calculate percentages for age groups
age_group_counts = reg.groupby(['AGE_GROUP', 'DONOR_TYPE']).size().unstack().fillna(0)
age_group_percentages = age_group_counts.div(age_group_counts.sum(axis=1), axis=0) * 100

stack_type=px.bar(age_group_percentages, text_auto='.2s')
st.plotly_chart(stack_type)


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

st.dataframe(area)
# Define steps for funnel
funnel_steps = ['Available Appointments', 'Appointments Attended', 'Successful Donations']

fig2 = go.Figure()

fig2.add_trace(go.Funnel(
    name = 'Canterbury',
    y=funnel_steps,
    x=[area.loc['CANTERBURY','Available_Slots'], area.loc['CANTERBURY','Appointments_Attended'], area.loc['CANTERBURY','Successful_Donations']],
     textinfo="value+percent previous+percent initial", hoverinfo='x+y+text+percent initial+percent previous'))

fig2.add_trace(go.Funnel(
    name = 'Cornwall',
    orientation = "h",
    y=funnel_steps,
    x=[area.loc['CORNWALL','Available_Slots'],  area.loc['CORNWALL','Appointments_Attended'], area.loc['CORNWALL','Successful_Donations']],
    textposition = "inside",
    textinfo="value+percent previous+percent initial", hoverinfo='x+y+text+percent initial+percent previous'))


fig2.add_trace(go.Funnel(
    name = 'Preston',
    orientation = "h",
    y=funnel_steps,
    x=[area.loc['PRESTON','Available_Slots'],area.loc['PRESTON','Appointments_Attended'], area.loc['PRESTON','Successful_Donations']],
    textposition = "inside",
    textinfo="value+percent previous+percent initial", hoverinfo='x+y+text+percent initial+percent previous'))

    
st.plotly_chart(fig2)

fig3=go.Figure()

fig3.add_trace(go.Funnel(
    name = 'Lewisham',
    orientation = "h",
    y=funnel_steps,
    x=[area.loc['LEWISHAM','Available_Slots'],area.loc['LEWISHAM','Appointments_Attended'], area.loc['LEWISHAM','Successful_Donations']],
    textposition = "inside",
    textinfo="value+percent previous+percent initial", hoverinfo='x+y+text+percent initial+percent previous'))

fig3.add_trace(go.Funnel(
    name = 'Durham',
    orientation = "h",
    y=funnel_steps,
    x=[area.loc['DURHAM','Available_Slots'],  area.loc['DURHAM','Appointments_Attended'], area.loc['DURHAM','Successful_Donations']],
    textposition = "inside",
    textinfo="value+percent previous+percent initial", hoverinfo='x+y+text+percent initial+percent previous'))


    
st.plotly_chart(fig3)
st.divider()
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

Ro_donor_age=px.bar(ro_donors_df['AGE_GROUP'])
Ro_donor_age.update_xaxes(categoryorder='array', categoryarray= ['<18', '18-30', '30-45', '45-60', '60+'])
st.plotly_chart(Ro_donor_age)

# st.subheader('Canterbury')

# # Create funnel plot for each area
# Cant_fun = go.Figure(go.Funnel(
#         y=funnel_steps,
#         x=[area.loc['CANTERBURY','Available_Slots'], area.loc['CANTERBURY','Appointments_Booked'], area.loc['CANTERBURY','Appointments_Attended'], area.loc['CANTERBURY','Successful_Donations']],
#         textinfo="value+percent previous+percent initial", hoverinfo='x+y+text+percent initial+percent previous'
#     ))


# Cant_fun.update_layout(title="Funnel Plot for Blood Donation Process in Canterbury")
# st.plotly_chart(Cant_fun)
# st.divider()
# st.subheader('Cornwall')

# # Create funnel plot for each area
# Corn_fun = go.Figure(go.Funnel(
#         y=funnel_steps,
#         x=[area.loc['CORNWALL','Available_Slots'], area.loc['CORNWALL','Appointments_Booked'], area.loc['CORNWALL','Appointments_Attended'], area.loc['CORNWALL','Successful_Donations']],
#         textinfo="value+percent previous+percent initial", hoverinfo='x+y+text+percent initial+percent previous'
#     ))


# Corn_fun.update_layout(title="Funnel Plot for Blood Donation Process in Cornwall")
# st.plotly_chart(Corn_fun)
# st.divider()
# st.subheader('Durham')

# # Create funnel plot for each area
# Durh_fun = go.Figure(go.Funnel(
#         y=funnel_steps,
#         x=[area.loc['DURHAM','Available_Slots'], area.loc['DURHAM','Appointments_Booked'], area.loc['DURHAM','Appointments_Attended'], area.loc['DURHAM','Successful_Donations']],
#         textinfo="value+percent previous+percent initial", hoverinfo='x+y+text+percent initial+percent previous'
#     ))


# Durh_fun.update_layout(title="Funnel Plot for Blood Donation Process in Durham")
# st.plotly_chart(Durh_fun)




# # ## Determine No. Appointments

# # ## Analysing Missed Appointments and Rejections

# # In[433]:


# plt.figure(figsize=(12, 6))
# sns.histplot(data=reg, x= 'NUM_DNAS', bins=20)
# plt.title('Distribution of Missed Appointments')
# plt.xlabel('Number of Missed Appointments')
# plt.ylabel('Frequency')
# plt.grid(True)
# plt.show()


# # In[23]:


# # Distribution of rejections
# plt.figure(figsize=(12, 6))
# sns.histplot(data= reg, x= 'NUM_REJECTIONS', bins=20)
# plt.title('Distribution of Rejections')
# plt.xlabel('Number of Rejections')
# plt.ylabel('Frequency')
# plt.grid(True)
# plt.show()


# # In[25]:


# # Impact of distance on missed appointments 
# distance_cols = ['DISTANCE_CORNWALL', 'DISTANCE_CANTERBURY', 'DISTANCE_PRESTON', 'DISTANCE_LEWISHAM', 'DISTANCE_DURHAM']
# fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# for i, col in enumerate(distance_cols):
#     sns.scatterplot(data=reg, x=col, y='NUM_DNAS', ax=axes[i//3, i%3])
#     axes[i//3, i%3].set_title(f'Impact of Distance on Donations: {col.split("_")[1]}')

# plt.tight_layout()
# plt.show()


# # ## Analysing 2 variables together 

# # In[168]:


# plt.figure(figsize=(15,8))
# blood_ethnicity = pd.crosstab(reg['BLOOD_GROUP'], reg['ETHNIC_GROUP'],normalize="index")
# sns.heatmap(blood_ethnicity, annot=True, fmt=".3f", cmap="YlGnBu")
# plt.title('Blood Group by Ethnicity')
# plt.show()


# # In[166]:


# blood_ethnicity.plot(kind='bar', 
#                     stacked=True, 
                    
#                     figsize=(12, 6))

# plt.legend(loc="upper left", ncol=2)
# plt.xlabel("Blood Group")
# plt.ylabel("Ethnic Proportion")
# plt.show()


# # In[177]:


# plt.figure(figsize=(15,8))
# dns_ethnicity = pd.crosstab(reg['NUM_DNAS'], reg['ETHNIC_GROUP'], normalize='index')
# sns.heatmap(dns_ethnicity, annot=True, fmt=".2f", cmap="YlGnBu")
# plt.title('Did Not Show by Ethnicity')
# plt.show()


# # In[179]:


# dns_ethnicity.plot(kind='bar', 
#                     stacked=True, 
                    
#                     figsize=(12, 6))

# plt.legend(loc="upper left", ncol=2)
# plt.xlabel("Did Not Show")
# plt.ylabel("Ethnic Proportion")
# plt.show()


# # In[435]:


# plt.figure(figsize=(15,8))
# dns_ethnicg = pd.crosstab(reg['NUM_DNAS'], reg['ETHNIC_GROUP'])
# sns.heatmap(dns_ethnicg, annot=True, fmt="d", cmap="YlGnBu")
# plt.title('Did Not Show by Ethnicity')
# plt.show()


# # ## Idea: Analyse Unknown blood types 

# # In[171]:


# plt.figure(figsize=(15,8))
# sns.jointplot(data=reg, x='BLOOD_GROUP', y='NUM_DONATIONS')

# plt.title('Blood Group and Number of donations ')
# plt.show()


# # In[271]:


# reg.fillna({'BLOOD_GROUP': 'Unknown'}, inplace=True)
# unknown_blood_type = reg[reg['BLOOD_GROUP'] == 'Unknown']


# outcome_counts = {
#     'Did Not Show': unknown_blood_type['NUM_DNAS'].sum(),
#     'Donate': unknown_blood_type['NUM_DONATIONS'].sum(),
#     'Rejected': unknown_blood_type['NUM_REJECTIONS'].sum(),
# }

# unknown_blood_type


# # In[268]:





# # In[267]:


# # Convert to DataFrame for plotting
# outcome_df = pd.DataFrame(list(outcome_counts.items()), columns=['Outcome', 'Count'])

# # Plot the data
# plt.figure(figsize=(10, 6))
# sns.barplot(data=outcome_df, x='Outcome', y='Count')
# plt.title('Outcomes for Unknown Blood Types')
# plt.ylabel('Count')
# plt.xlabel('Outcome')
# plt.show()


# # In[273]:


# eth=reg.groupby('ETHNICITY', group_keys=True)[['NUM_DNAS']].apply(lambda x: x.sum())
# eth


# # In[283]:


# reg.groupby('ETHNICITY', group_keys=True).apply(lambda x: x.count())


# # ## Slot Availability and Utilization 

# # In[32]:


# # Count available slots by donor center and month
# slots_by_center = slots.groupby([slots['SESDATE'].dt.to_period('M'), 'DC']).size().unstack(fill_value=0)

# # Plot availability
# slots_by_center.plot(kind='line', figsize=(12, 6))
# plt.title('Available Slots Over Time by Donor Center')
# plt.xlabel('Month')
# plt.ylabel('Number of Available Slots')
# plt.grid(True)
# plt.show()


# # In[88]:


# slots.set_index('SESDATE').resample('W').size().plot(title='Slots Over Time', markersize=10, marker='o')


# # In[87]:


# slots_by_center = slots.groupby([slots['SESDATE'].dt.to_period('W'), 'DC']).size().unstack(fill_value=0)

# #slots_by_center.plot(kind='line',title='Slots Over Time', markersize=8, marker='o', figsize=(12, 6))


# pts= slots_by_center.resample('W').asfreq()

# fig, ax = plt.subplots(figsize=(12, 12))
# slots_by_center.plot(kind='line', ax=ax)
# pts.plot(marker='o', ax=ax, lw=0, color='black', legend=False)


# pad = 10
# for idx, val in pts.stack().items():
#     ax.annotate(val, (idx[0], val+pad))

# ax.grid(axis='y')

# for row, x in enumerate(slots_by_center.index):
#     pass
#     #print(row)
#     #print(x)
#     #plt.text(x,df2[row],df2[str(row)], fontsize=12, bbox=dict(facecolor='blue', alpha=0.1), ha= 'left', va= 'baseline')



# # In[115]:


# # Facet grid for ethnicity, blood group, and number of donations
# g = sns.FacetGrid(reg, col="BLOOD_GROUP", row="ETHNIC_GROUP", margin_titles=True, height=4)
# g.map(sns.boxplot, "NUM_DONATIONS")
# g.add_legend()
# plt.show()


# # In[91]:


# from mpl_toolkits.mplot3d import Axes3D

# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(reg['DISTANCE_DURHAM'], reg['AGE'], reg['NUM_DONATIONS'])
# ax.set_xlabel('Distance to Durham (miles)')
# ax.set_ylabel('Age')
# ax.set_zlabel('Number of Donations')
# plt.title('3D Scatter Plot: Distance, Age, and Number of Donations')
# plt.show()


# # In[127]:


# from pandas.plotting import scatter_matrix 
  
# # selecting three numerical features 
# features = ['AGE', 'DISTANCE_DURHAM', 'NUM_DONATIONS'] 
   
# # plotting the scatter matrix 
# # with the features 
# scatter_matrix(reg[features]) 
# plt.show() 


# # In[126]:


# reg


# # In[ ]:




# # In[ ]:





# # In[ ]:




