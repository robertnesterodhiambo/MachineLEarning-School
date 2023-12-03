import excel "C:\Users\neste\Downloads\P_Data_Extract_From_World_Development_Indicators.xlsx", sheet("Data")
browse
import excel "C:\Users\neste\Downloads\P_Data_Extract_From_World_Development_Indicators.xlsx", sheet("Data") firstrow case(lower) clear
browse
save "C:\Users\neste\Downloads\P_Data_Extract_From_World_Development_Indicators.dta"
twoway (scatter yr2012 yr2013) (scatter yr2013 yr2014) (scatter yr2014 yr2015) (scatter yr2015 yr2016)
pwcorr
import excel "C:\Users\neste\Downloads\P_Data_Extract_From_World_Development_Indicators.xlsx", sheet("Data") firstrow clear
twoway (scatter GDPpercapitacurrentUS GDPpercapitagrowthannual)
twoway (line GDPpercapitacurrentUS Populationgrowthannual)
twoway (line GDPpercapitacurrentUS Populationgrowthannual, sort)
pwcorr
summarize GDPpercapitacurrentUS GDPpercapitagrowthannual Populationgrowthannual Inflationconsumerpricesannu Netinvestmentinnonfinanciala Finalconsumptionexpenditure Centralgovernmentdebttotal, detail
regress GDPpercapitacurrentUS GDPpercapitagrowthannual Populationgrowthannual Inflationconsumerpricesannu Netinvestmentinnonfinanciala Finalconsumptionexpenditure Centralgovernmentdebttotal, tsscons
regress GDPpercapitacurrentUS GDPpercapitagrowthannual Populationgrowthannual Inflationconsumerpricesannu Netinvestmentinnonfinanciala Finalconsumptionexpenditure Centralgovernmentdebttotal
