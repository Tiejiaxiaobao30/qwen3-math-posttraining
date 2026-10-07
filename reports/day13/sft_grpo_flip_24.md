# Day13｜SFT → GRPO 正确性翻转案例审计

固定 dev500：SFT 390/500 (78.00%) → GRPO 392/500 (78.40%)。

- Improved: 13
- Regressed: 11
- Net gain: 2

## Case 1｜IMPROVED｜ID 121

**Transition:** `wrong -> correct`

**Gold:** `6`

**Question**

Stella and Twinkle are filling up a truck with a capacity of 6000 stone blocks at the rate of 250 blocks per hour per person. They work for four hours and are then joined by 6 other people who also work at the same rate. How many hours did filling the truck take?

### SFT

- status: `wrong`
- parsed: `6.66666666667`
- tokens: 194

```text
Stella and Twinkle working together fill up 2*250 = 500 blocks per hour.
After four hours, they have filled up 500*4 = 2000 blocks.
The remaining blocks to be filled are 6000-2000 = 4000 blocks.
The combined rate of the 6 people and Stella and Twinkle is 250*6 = 1500 blocks per hour.
The remaining blocks will be filled in 4000/1500 = 2.66666666667 hours.
The total time to fill the truck is 4+2.66666666667 = 6.66666666667 hours.
Final answer: 6.66666666667
```

### GRPO

- status: `correct`
- parsed: `6`
- tokens: 189

```text
Stella and Twinkle working together fill up 2*250=500 blocks per hour.
After working for 4 hours, they have filled up 500*4=2000 blocks.
The remaining blocks to be filled are 6000-2000=4000 blocks.
The 6 people working with Stella and Twinkle fill up 6*250=1500 blocks per hour.
The 6 people and Stella and Twinkle working together fill up 1500+500=2000 blocks per hour.
The 6 people and Stella and Twinkle working together will fill up the remaining 4000 blocks in 4000/2000=2 hours.
The total time to fill up the truck is 4+2=6 hours.
Final answer: 6
```

### Manual diagnosis

- Category: TODO
- Note: TODO

---

## Case 2｜IMPROVED｜ID 356

**Transition:** `wrong -> correct`

**Gold:** `66`

**Question**

Students at Highridge High earn 2 points for each correct answer during a quiz bowl If a student correctly answers all the questions in a round, the student is awarded an additional 4 point bonus. They played a total of five rounds each consisting of five questions. If James only missed one question, how many points did he get?

### SFT

- status: `wrong`
- parsed: `54`
- tokens: 60

```text
James answered 5*5=25 questions
He got 25*2=50 points
He got a 4 point bonus because he answered all the questions in one round
So he got 50+4=54 points
Final answer: 54
```

### GRPO

- status: `correct`
- parsed: `66`
- tokens: 62

```text
James answered 5*5=25 questions
He got 25*2=50 points for the correct answers
He got 4*4=16 points for the bonus
So he got 50+16=66 points
Final answer: 66
```

### Manual diagnosis

- Category: TODO
- Note: TODO

---

## Case 3｜IMPROVED｜ID 1275

**Transition:** `wrong -> correct`

**Gold:** `650`

**Question**

Ginger is weaving mats out of colored straw. Each mat takes 20 red straws, 30 orange straws, and half as many green straws as orange straws. How many straws does she need to make 10 mats?

### SFT

- status: `wrong`
- parsed: `550`
- tokens: 59

```text
Ginger needs 20 + 30 + 30/2 = 55 straws to make one mat.
So she needs 55 x 10 = 550 straws to make 10 mats.
Final answer: 550
```

### GRPO

- status: `correct`
- parsed: `650`
- tokens: 70

```text
Ginger needs 20 red straws + 30 orange straws + 30/2 green straws = 65 straws for one mat.
So she needs 65 straws x 10 mats = 650 straws to make 10 mats.
Final answer: 650
```

### Manual diagnosis

- Category: TODO
- Note: TODO

---

## Case 4｜IMPROVED｜ID 4578

**Transition:** `wrong -> correct`

**Gold:** `6`

**Question**

Grace just started her own business. Each week, she charges 300 dollars. Grace's client will pay her every 2 weeks. How many weeks will it take for Grace to get 1800 dollars?

### SFT

- status: `wrong`
- parsed: `3`
- tokens: 53

```text
Grace will get 300 x 2 = 600 dollars every 2 weeks.
Grace will get 1800 / 600 = 3 weeks to get 1800 dollars.
Final answer: 3
```

### GRPO

- status: `correct`
- parsed: `6`
- tokens: 98

```text
Grace's client will pay her every 2 weeks, so she will get paid 300 x 2 = 600 dollars every 2 weeks.
Grace needs to get 1800 dollars, so she will need 1800 / 600 = 3 sets of 2 weeks.
Therefore, it will take Grace 3 x 2 = 6 weeks to get 1800 dollars.
Final answer: 6
```

### Manual diagnosis

- Category: TODO
- Note: TODO

---

## Case 5｜IMPROVED｜ID 4715

**Transition:** `wrong -> correct`

**Gold:** `102`

**Question**

Mary is paying her monthly garbage bill for a month with exactly four weeks. The garbage company charges Mary $10 per trash bin and $5 per recycling bin every week, and Mary has 2 trash bins and 1 recycling bin. They're giving her an 18% discount on the whole bill before fines for being elderly, but also charging her a $20 fine for putting inappropriate items in a recycling bin. How much is Mary's garbage bill?

### SFT

- status: `wrong`
- parsed: `40.5`
- tokens: 133

```text
Mary's trash bin bill is $10 x 2 = $20.
Mary's recycling bin bill is $5 x 1 = $5.
Mary's garbage bill before the discount and the fine is $20 + $5 = $25.
Mary's discount is $25 x 18% = $4.50.
Mary's garbage bill after the discount is $25 - $4.50 = $20.50.
Mary's garbage bill after the fine is $20.50 + $20 = $40.50.
Final answer: 40.5
```

### GRPO

- status: `correct`
- parsed: `102`
- tokens: 181

```text
Mary's trash bins cost 2 trash bins * $10 trash bin = $20 per week.
Mary's recycling bin costs 1 recycling bin * $5 recycling bin = $5 per week.
Mary's garbage bill is $20 trash bin + $5 recycling bin = $25 per week.
Mary's garbage bill for the month is $25 per week * 4 weeks = $100.
Mary gets an 18% discount on the garbage bill, so the discount is $100 * 18% = $18.
Mary's garbage bill after the discount is $100 - $18 = $82.
Mary gets a $20 fine for putting inappropriate items in a recycling bin, so her garbage bill is $82 + $20 = $102.
Final answer: 102
```

### Manual diagnosis

- Category: TODO
- Note: TODO

---

## Case 6｜IMPROVED｜ID 5247

**Transition:** `wrong -> correct`

**Gold:** `60`

**Question**

Grant spends $200.00 a year to have the newspaper delivered daily to his house.  Juanita buys the newspaper daily.  Monday through Saturday, she spends $0.50 and on Sunday she spends $2.00.  How much more money does Juanita spend buying the newspaper yearly than Grant?

### SFT

- status: `wrong`
- parsed: `1260`
- tokens: 181

```text
Grant spends $200.00 a year on the newspaper.
Juanita spends $0.50 a day for 6 days and $2.00 for Sunday, so she spends $0.50*6 + $2.00 = $4.00 a day.
Juanita spends $4.00 a day for 365 days, so she spends $4.00*365 = $1460.00 a year on the newspaper.
Juanita spends $1460.00 a year on the newspaper and Grant spends $200.00 a year, so Juanita spends $1460.00 - $200.00 = $1260.00 more on the newspaper than Grant.
Final answer: 1260
```

### GRPO

- status: `correct`
- parsed: `60`
- tokens: 183

```text
Grant spends $200.00 a year on the newspaper.
Juanita spends $0.50 a day for 6 days, so she spends 6*$0.50 = $3.00 a week on the newspaper.
Juanita spends $2.00 a day on Sunday, so she spends 1*$2.00 = $2.00 on Sunday.
Juanita spends $3.00 a week + $2.00 on Sunday = $5.00 a week on the newspaper.
Juanita spends $5.00 a week * 52 weeks = $260.00 a year on the newspaper.
Juanita spends $260.00 a year - $200.00 a year = $60.00 more than Grant.
Final answer: 60
```

### Manual diagnosis

- Category: TODO
- Note: TODO

---

## Case 7｜IMPROVED｜ID 5588

**Transition:** `wrong -> correct`

**Gold:** `22`

**Question**

Alice's white water rafting class is composed of 40 students and 10 instructors, including herself.  She has 20 life vests on hand.  20% of her students are bringing life vests.  How many more life vests will Alice need to get so that entire class has one?

### SFT

- status: `wrong`
- parsed: `20`
- tokens: 99

```text
Alice has 40 students + 10 instructors = 50 people in her class.
20% of 50 people is 20/100 * 50 = 10 people.
So Alice needs 50 - 10 = 40 more life vests.
Alice has 20 life vests on hand, so she needs 40 - 20 = 20 more life vests.
Final answer: 20
```

### GRPO

- status: `correct`
- parsed: `22`
- tokens: 90

```text
Alice has 40 students + 10 instructors = 50 people in her class.
20% of her students are bringing life vests, so 20/100 * 40 = 8 students are bringing life vests.
Alice has 20 life vests on hand, so she needs 50 - 20 - 8 = 22 more life vests.
Final answer: 22
```

### Manual diagnosis

- Category: TODO
- Note: TODO

---

## Case 8｜IMPROVED｜ID 5806

**Transition:** `wrong -> correct`

**Gold:** `2`

**Question**

Dylan filled his glass with ice cubes before he poured his soda into it. He counted and there were 8 ice cubes in his glass. Later when he prepares a pitcher of lemonade he puts two times as many ice cubes in the pitcher. Then it is time to refill the ice cube trays which have 12 spaces each for ice cubes. How many trays does Dylan need to fill if he used exactly all the ice they had in them?

### SFT

- status: `wrong`
- parsed: `6`
- tokens: 73

```text
Dylan used 8 x 2 = 16 ice cubes for the pitcher.
He used 8 + 16 = 24 ice cubes.
He has 12 x 12 = 144 spaces in the trays.
He needs 144 / 24 = 6 trays.
Final answer: 6
```

### GRPO

- status: `correct`
- parsed: `2`
- tokens: 79

```text
Dylan used 8 x 2 = 16 ice cubes for the pitcher of lemonade.
He used 8 + 16 = 24 ice cubes in total.
He used 12 x 2 = 24 ice cubes to fill 2 trays.
Dylan needs 2 more trays to fill up all the ice cubes.
Final answer: 2
```

### Manual diagnosis

- Category: TODO
- Note: TODO

---

## Case 9｜IMPROVED｜ID 5857

**Transition:** `wrong -> correct`

**Gold:** `1000`

**Question**

In one day, Ella's dog eats 4 pounds of food for every one pound of food that Ella eats. How much food do Ella and her dog in 10 days if Ella eat 20 pounds of food each day?

### SFT

- status: `wrong`
- parsed: `50`
- tokens: 73

```text
In one day, Ella's dog eats 4 pounds of food for every one pound of food that Ella eats, so in one day, Ella and her dog eat 4+1=5 pounds of food.
In 10 days, Ella and her dog eat 5*10=50 pounds of food.
Final answer: 50
```

### GRPO

- status: `correct`
- parsed: `1000`
- tokens: 85

```text
In 10 days, Ella eats 20*10 = 200 pounds of food.
In 10 days, her dog eats 4*200 = 800 pounds of food.
In 10 days, Ella and her dog eat 200+800 = 1000 pounds of food.
Final answer: 1000
```

### Manual diagnosis

- Category: TODO
- Note: TODO

---

## Case 10｜IMPROVED｜ID 6339

**Transition:** `wrong -> correct`

**Gold:** `75`

**Question**

Sofia and her mother go to the grocery store and buys 10kgs of oranges to be used in their home for the week. While at the store, Sofia's father calls and says they're to add 5 more kgs since their neighbor also wanted some supplies. When they reach home, they estimated that for the next coming two weeks they'll have to buy twice as many oranges each week for the supplies to be enough. What the total quantity of oranges they would have bought after the three weeks.

### SFT

- status: `wrong`
- parsed: `45`
- tokens: 79

```text
For the first week, they bought 10 + 5 = 15 kgs of oranges.
For the next two weeks, they would have bought 15 x 2 = 30 kgs of oranges.
For the three weeks, they would have bought 15 + 30 = 45 kgs of oranges.
Final answer: 45
```

### GRPO

- status: `correct`
- parsed: `75`
- tokens: 91

```text
The total quantity of oranges they bought in the first week is 10 + 5 = 15 kgs.
For the next two weeks, they would have bought 15 x 2 = 30 kgs of oranges each week.
The total quantity of oranges they would have bought after the three weeks is 15 + 30 + 30 = 75 kgs.
Final answer: 75
```

### Manual diagnosis

- Category: TODO
- Note: TODO

---

## Case 11｜IMPROVED｜ID 6348

**Transition:** `wrong -> correct`

**Gold:** `18`

**Question**

Jill runs a business breeding llamas. Nine of her llamas got pregnant with 1 calf, and 5 of them got pregnant with twins. After they give birth, Jill traded 8 of the calves for 2 new adult llamas. Then she sells 1/3 of her herd at the market. How many total llamas  (calves and adults) does Jill have now?

### SFT

- status: `wrong`
- parsed: `9`
- tokens: 117

```text
Jill has 9*1=9 calves from the single calves.
She has 5*2=10 calves from the twins.
She has 9+10=19 calves.
She has 19+9+5=33 llamas.
She trades 8 of the calves for 2 new adult llamas, so she has 33-8+2=27 llamas.
She sells 1/3 of her herd, so she has 27/3=9 llamas left.
Final answer: 9
```

### GRPO

- status: `correct`
- parsed: `18`
- tokens: 131

```text
Jill has 9*1=9 calves from the single pregnancies.
She has 5*2=10 calves from the twin pregnancies.
She has 9+10=19 calves.
She has 19+9+5=33 llamas.
She trades 8 of the calves for 2 new llamas, so she has 33-8+2=27 llamas.
She sells 1/3 of her herd, so she sells 27/3=9 llamas.
She has 27-9=18 llamas left.
Final answer: 18
```

### Manual diagnosis

- Category: TODO
- Note: TODO

---

## Case 12｜IMPROVED｜ID 6699

**Transition:** `wrong -> correct`

**Gold:** `4100`

**Question**

Bob has to hire someone to fix his garden.  A storm destroyed all 20 of his rose bushes.  He decides to replant all the rose bushes.  Each rose bush costs $150.  He also needs to pay a gardener $30 an hour, 5 hours each day for 4 days.  The final expense is 100 cubic feet of soil sold for $5 per cubic foot.  How much did the entire gardening project cost?

### SFT

- status: `wrong`
- parsed: `3100`
- tokens: 79

```text
The rose bushes cost 20*150=$3000
The gardener cost 30*5*4=$600
The soil cost 100*5=$500
The total cost is 3000+600+500=$3100
Final answer: 3100
```

### GRPO

- status: `correct`
- parsed: `4100`
- tokens: 79

```text
The rose bushes cost 20*150=$3000
The gardener cost 30*5*4=$600
The soil cost 100*5=$500
The total cost is 3000+600+500=$4100
Final answer: 4100
```

### Manual diagnosis

- Category: TODO
- Note: TODO

---

## Case 13｜IMPROVED｜ID 6713

**Transition:** `wrong -> correct`

**Gold:** `18`

**Question**

Charlotte is making a pie. She can't decide if she will make cherry or blueberry. So she decides to make whichever pie is cheapest. Both pies will need $2 for flour, $1 for sugar and $1.5 for eggs and butter to make the crust. The blueberry pie needs 3 pounds of blueberries. The blueberries come in 8 oz containers for $2.25. For the cherry pie, she needs 4 pounds and they come in a four-pound bag for $14. What is the total price to make the cheapest pie?

### SFT

- status: `wrong`
- parsed: `5.06`
- tokens: 116

```text
The crust will cost $2+$1+$1.50=$4.50
The blueberry pie needs 3 pounds/16 oz/pound=18/8=2.25 pounds of blueberries
The blueberry pie will cost $2.25*2.25=$5.06
The cherry pie needs 4 pounds/4 pounds/bag=$1
The cherry pie will cost $14
The blueberry pie is cheaper at $5.06
Final answer: 5.06
```

### GRPO

- status: `correct`
- parsed: `18`
- tokens: 131

```text
The crust will cost $2+$1+$1.50=$4.50
The blueberry pie needs 3 pounds*16 oz/pound=48 oz of blueberries
The blueberry pie needs 48 oz/8 oz/container=6 containers of blueberries
The blueberry pie will cost 6*2.25=$13.50
The cherry pie will cost 14$ because it is the only one that is cheaper
The total cost to make the cheapest pie is 13.50+4.50=$18
Final answer: 18
```

### Manual diagnosis

- Category: TODO
- Note: TODO

---

## Case 14｜REGRESSED｜ID 370

**Transition:** `correct -> wrong`

**Gold:** `64`

**Question**

Nurse Missy is attending to the needs of 12 patients in her hospital ward.  Most of her patients require standard care, but one-third of her patients have special dietary requirements, which increases the serving time by 20%.  At dinner time, she brings each patient their meal. It takes 5 minutes to serve each standard care patient.  How long does it take, in minutes, for Missy to serve dinner to all of her patients?

### SFT

- status: `correct`
- parsed: `64`
- tokens: 97

```text
Missy has 12/3=4 patients with special dietary requirements.
The 4 patients with special dietary requirements will take 4*5*1.2=24 minutes to serve.
The remaining 12-4=8 patients will take 8*5=40 minutes to serve.
Thus, it will take Missy 24+40=64 minutes to serve dinner to all of her patients.
Final answer: 64
```

### GRPO

- status: `wrong`
- parsed: `60`
- tokens: 102

```text
Missy has 12/3=4 patients with special dietary requirements.
The 4 patients with special dietary requirements take 4*5=20 minutes to serve.
The remaining 12-4=8 patients require standard care.
The 8 patients with standard care take 8*5=40 minutes to serve.
In total, it takes Missy 20+40=60 minutes to serve dinner to all of her patients.
Final answer: 60
```

### Manual diagnosis

- Category: TODO
- Note: TODO

---

## Case 15｜REGRESSED｜ID 495

**Transition:** `correct -> wrong`

**Gold:** `22`

**Question**

MIlle is making snack packs for her kindergarten class. She's going to divide 64 pretzels, four times as many goldfish, and 32 suckers into baggies for the 16 kids in the class. How many items does each baggie have?

### SFT

- status: `correct`
- parsed: `22`
- tokens: 117

```text
First find the total number of goldfish: 64 pretzels * 4 = 256 goldfish
Then add the number of each type of snack to find the total number of snacks: 256 goldfish + 64 pretzels + 32 suckers = 352 snacks
Then divide the total number of snacks by the number of kids to find the number of snacks per baggie: 352 snacks / 16 kids = 22 snacks per baggie
Final answer: 22
```

### GRPO

- status: `wrong`
- parsed: `20`
- tokens: 87

```text
Mille has 64 pretzels, 4 times as many goldfish, and 32 suckers, for a total of 64 + 4*64 + 32 = 320 items.
She wants to divide these items into 16 baggies, so each baggie has 320/16 = 20 items.
Final answer: 20
```

### Manual diagnosis

- Category: TODO
- Note: TODO

---

## Case 16｜REGRESSED｜ID 765

**Transition:** `correct -> wrong`

**Gold:** `135`

**Question**

Brenda picks 250 peaches. When she sorts through them, only 60% are fresh, and Brenda has to throw 15 away for being too small. How many peaches does Brenda have left?

### SFT

- status: `correct`
- parsed: `135`
- tokens: 52

```text
Brenda has 250 x 60/100 = 150 fresh peaches.
She has 150 - 15 = 135 peaches left.
Final answer: 135
```

### GRPO

- status: `wrong`
- parsed: `235`
- tokens: 52

```text
Brenda has 250 x 60/100 = 150 fresh peaches.
She has 250 - 15 = 235 peaches left.
Final answer: 235
```

### Manual diagnosis

- Category: TODO
- Note: TODO

---

## Case 17｜REGRESSED｜ID 850

**Transition:** `correct -> wrong`

**Gold:** `90`

**Question**

Viviana has five more chocolate chips than Susana, while Susana has 3/4 as many vanilla chips as Viviana. If Viviana has 20 Vanilla chips and Susana 25 chocolate chips, calculate the total number of chips they have together.

### SFT

- status: `correct`
- parsed: `90`
- tokens: 147

```text
Viviana has 5 more chocolate chips than Susana, who has 25 chocolate chips, so Viviana has 25+5 = 30 chocolate chips.
Susana has 3/4 as many vanilla chips as Viviana, who has 20 vanilla chips, so Susana has 3/4*20 = 15 vanilla chips.
Together, Viviana and Susana have 20+15 = 35 vanilla chips.
Viviana and Susana have 30+25 = 55 chocolate chips.
In total, Viviana and Susana have 35+55 = 90 chips.
Final answer: 90
```

### GRPO

- status: `wrong`
- parsed: `107.5`
- tokens: 81

```text
Viviana has 20+5=25 chocolate chips.
Susana has 25*3/4=18.75 vanilla chips.
The total number of chips they have together is 20+25+18.75+25+18.75=107.5
Final answer: 107.5
```

### Manual diagnosis

- Category: TODO
- Note: TODO

---

## Case 18｜REGRESSED｜ID 2995

**Transition:** `correct -> wrong`

**Gold:** `720`

**Question**

Amiyah is cutting some trees to build a cow shade. For every tree she cuts, she plants 5 new trees. If there were 400 trees on her farm and she cut 20% of them, calculate the total number of trees on the farm.

### SFT

- status: `correct`
- parsed: `720`
- tokens: 129

```text
If there were 400 trees on her farm and she cut 20% of them, she cut 400*20/100 = 80 trees.
If she cut 80 trees, there are 400-80 = 320 trees left on the farm.
For every tree she cuts, she plants 5 new trees, so she planted 80*5 = 400 new trees.
The total number of trees on the farm is 320+400 = 720 trees.
Final answer: 720
```

### GRPO

- status: `wrong`
- parsed: `800`
- tokens: 96

```text
If there were 400 trees on her farm and she cut 20% of them, she cut 400 x 20/100 = 80 trees.
The total number of trees that she cut is 80 x 5 = 400 trees.
The total number of trees on the farm is 400 + 400 = 800 trees.
Final answer: 800
```

### Manual diagnosis

- Category: TODO
- Note: TODO

---

## Case 19｜REGRESSED｜ID 3088

**Transition:** `correct -> wrong`

**Gold:** `14`

**Question**

Keanu's motorcycle can store 8 liters of gasoline. If his destination is 280 miles away and his motorcycle consumes 8 liters of gasoline per 40 miles, how many times does Keanu have to refill his motorcycle with gasoline if he will make a round trip?

### SFT

- status: `correct`
- parsed: `14`
- tokens: 129

```text
Keanu's motorcycle consumes 8 liters of gasoline per 40 miles, so it consumes 8/40 = 0.2 liters of gasoline per mile.
The total distance for a round trip is 280 x 2 = 560 miles.
So, Keanu's motorcycle will consume 560 x 0.2 = 112 liters of gasoline for a round trip.
Since his motorcycle can only store 8 liters of gasoline, Keanu will have to refill his motorcycle 112/8 = 14 times.
Final answer: 14
```

### GRPO

- status: `wrong`
- parsed: `7`
- tokens: 107

```text
Keanu's motorcycle consumes 8 liters of gasoline per 40 miles, so it consumes 8/40 = 0.2 liters of gasoline per mile.
If his destination is 280 miles away, Keanu's motorcycle will consume 280 * 0.2 = 56 liters of gasoline.
Since his motorcycle can store 8 liters of gasoline, Keanu will have to refill his motorcycle 56/8 = 7 times.
Final answer: 7
```

### Manual diagnosis

- Category: TODO
- Note: TODO

---

## Case 20｜REGRESSED｜ID 3565

**Transition:** `correct -> format_fail`

**Gold:** `40`

**Question**

Everett built a rectangular concrete patio that was four times as long as it was wide.  If the perimeter of the patio was 100 feet, what was the length of the patio, in feet?

### SFT

- status: `correct`
- parsed: `40`
- tokens: 103

```text
Let x be the width of the patio.
The length of the patio is 4x.
The perimeter of the patio is 2(x+4x)=100
The perimeter of the patio is 2(5x)=100
The perimeter of the patio is 10x=100
The width of the patio is x=10
The length of the patio is 4x=4(10)=40
Final answer: 40
```

### GRPO

- status: `format_fail`
- parsed: `None`
- tokens: 95

```text
Let x be the width of the patio.
The length of the patio is 4x.
The perimeter of the patio is 2(x+4x)=100
6x=100
x=100/6=50/3
The length of the patio is 4(50/3)=200/3=66 2/3 feet
Final answer: 66 2/3
```

### Manual diagnosis

- Category: TODO
- Note: TODO

---

## Case 21｜REGRESSED｜ID 4666

**Transition:** `correct -> wrong`

**Gold:** `25`

**Question**

Dana has 15 more pencils than Jayden, who has twice as much as Marcus. How many more pencils does Dana have than Marcus, if Jayden has 20 pencils?

### SFT

- status: `correct`
- parsed: `25`
- tokens: 50

```text
Marcus has 20/2=10 pencils.
Dana has 20+15=35 pencils.
Dana has 35-10=25 more pencils than Marcus.
Final answer: 25
```

### GRPO

- status: `wrong`
- parsed: `15`
- tokens: 50

```text
Marcus has 20/2=10 pencils.
Dana has 10+15=25 pencils.
Dana has 25-10=15 more pencils than Marcus.
Final answer: 15
```

### Manual diagnosis

- Category: TODO
- Note: TODO

---

## Case 22｜REGRESSED｜ID 5382

**Transition:** `correct -> wrong`

**Gold:** `45`

**Question**

Emily is on a flight from New York to Hawaii that is 10 hours long. To kill time, she watches 3 TV episodes that are each 25 minutes long. Then she sleeps for 4 and a half hours. Then she watches 2 movies that are each an hour and 45 minutes long. After doing all these things, assuming there is no break time between activities, how many more minutes are left in the flight before Emily reaches Hawaii.

### SFT

- status: `correct`
- parsed: `45`
- tokens: 198

```text
First find the total time Emily spends watching TV episodes: 3 episodes * 25 minutes/episode = 75 minutes
Then find the total time Emily spends sleeping: 4.5 hours * 60 minutes/hour = 270 minutes
Then find the total time Emily spends watching movies: 2 movies * 105 minutes/movie = 210 minutes
Then add the time spent on each activity to find the total time spent: 75 minutes + 270 minutes + 210 minutes = 555 minutes
Then subtract the time spent on activities from the total flight time to find the remaining time: 10 hours * 60 minutes/hour = 600 minutes
Then subtract the time spent on activities from the total flight time to find the remaining time: 600 minutes - 555 minutes = 45 minutes
Final answer: 45
```

### GRPO

- status: `wrong`
- parsed: `52.5`
- tokens: 182

```text
First find the total time Emily spends watching TV episodes: 25 minutes/episode * 3 episodes = 75 minutes
Then find the total time Emily spends sleeping: 4.5 hours * 60 minutes/hour = 270 minutes
Then find the total time Emily spends watching movies: 1 hour 45 minutes/movie * 2 movies = 3 hours 45 minutes
Then add up the time spent on each activity to find the total time spent: 75 minutes + 270 minutes + 202.5 minutes = 547.5 minutes
Then subtract the total time spent from the total flight time to find the remaining time: 10 hours * 60 minutes/hour - 547.5 minutes = 52.5 minutes
Final answer: 52.5
```

### Manual diagnosis

- Category: TODO
- Note: TODO

---

## Case 23｜REGRESSED｜ID 7040

**Transition:** `correct -> wrong`

**Gold:** `5`

**Question**

There are 20 dolphins in the aqua park. One-fourth of the dolphins are fully trained. Two-third of the remaining dolphins are currently in training and the rest will be trained next month. How many dolphins will be trained next month?

### SFT

- status: `correct`
- parsed: `5`
- tokens: 80

```text
There are 20/4=5 fully trained dolphins.
So, there are 20-5=15 dolphins that are not fully trained.
Two-third of the remaining dolphins are currently in training, which is 15*2/3=10 dolphins.
So, 15-10=5 dolphins will be trained next month.
Final answer: 5
```

### GRPO

- status: `wrong`
- parsed: `10`
- tokens: 73

```text
There are 20/4=5 fully trained dolphins.
So, there are 20-5=15 dolphins that are not fully trained.
There are 15/3=5 dolphins currently in training.
So, there are 15-5=10 dolphins that will be trained next month.
Final answer: 10
```

### Manual diagnosis

- Category: TODO
- Note: TODO

---

## Case 24｜REGRESSED｜ID 7207

**Transition:** `correct -> wrong`

**Gold:** `95`

**Question**

If the first skyscraper was built 100 years ago, how many years in the future will it be 5 years before its 200th anniversary of being built?

### SFT

- status: `correct`
- parsed: `95`
- tokens: 92

```text
The 200th anniversary of the first skyscraper will be 200 years from now.
5 years before the 200th anniversary will be 200 - 5 = 195 years from now.
Since the first skyscraper was built 100 years ago, it will be 195 - 100 = 95 years in the future.
Final answer: 95
```

### GRPO

- status: `wrong`
- parsed: `195`
- tokens: 72

```text
The 200th anniversary of the first skyscraper will be 200 years from now.
5 years before the 200th anniversary will be 200-5 = 195 years from now.
The first skyscraper will be 195 years in the future.
Final answer: 195
```

### Manual diagnosis

- Category: TODO
- Note: TODO

---

