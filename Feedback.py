def feedback(percent): 
    if (percent < 30): 
        return ("Your risk level is **LOW**. ")

    elif (percent < 60): 
        return ("Your risk level is **MEDIUM**. You may consider seeking medical attention if you experience these similar symptoms a few more days.")
        
    elif (percent <= 100):
        return ("Your risk level is **HIGH**. You are advised to seek immediate medical attention.")
    else: 
        return('error')