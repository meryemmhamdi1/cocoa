
if __name__=="__main__":
    with open("logs/sup_cs/encdec/test_cs_0.txt") as file:
        data = file.readlines()

    for line in data:
        if "BEFORE REMOVING CS TAGS " in line:
            parts = line.split("[")
            print("encoding:", parts[1].split(",")[0], "decoding:", parts[2].split(",")[0])
