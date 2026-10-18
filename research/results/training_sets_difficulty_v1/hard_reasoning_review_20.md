# Selected hard solutions for reasoning review

Seeded review: 15 selected 1/4 problems, five selected 2–3/4 problems. These are the exact solutions proposed for the hard training set. No examples have been removed and no training has run.

## 1. Shuffled index 1530 — 1/4; candidate 4

**Question**
In Fred the Farmer's barn, there were 100 horses, 29 sheep, and 9 chickens. Brian bought half of the animals and sold them at the market. Jeremy then gifted Fred an additional 37 goats. If exactly half of the animals in the barn are male animals, how many male animals are there in the barn?

**Reference answer:** 53
**Solution tokens:** 138 (+1 EOS supervised); full sequence 651

**Selected model solution**
```text
There were 100 + 29 + 9 = 138 animals in the barn initially.
Brian bought half of the animals, so he bought 138/2 = 69 animals.
After Brian bought the animals, there were 138 - 69 = 69 animals left in the barn.
Jeremy gifted Fred an additional 37 goats, so there were 69 + 37 = 106 animals in the barn.
Since exactly half of the animals in the barn are male animals, there are 106/2 = 53 male animals in the barn.
#### 53
```

**Human review:** sound / substantive error / uncertain. Notes:

## 2. Shuffled index 1074 — 1/4; candidate 3

**Question**
John buys 20 hours of music a month.  The average length of a song is 3 minutes.  He buys each song for $.50.  How much does he pay for music a year?

**Reference answer:** 2400
**Solution tokens:** 87 (+1 EOS supervised); full sequence 572

**Selected model solution**
```text
20 hours a month is 20*60=1200 minutes.
So, he buys 1200/3=400 songs a month.
Therefore, he buys 400*12=4800 songs a year.
So, he pays 4800*$.50=$2400 a year.
#### 2400
```

**Human review:** sound / substantive error / uncertain. Notes:

## 3. Shuffled index 1221 — 1/4; candidate 1

**Question**
James does chores around the class.  There are 3 bedrooms, 1 living room, and 2 bathrooms to clean.  The bedrooms each take 20 minutes to clean.  The living room takes as long as the 3 bedrooms combined.  The bathroom takes twice as long as the living room.  He also cleans the outside which takes twice as long as cleaning the house.  He splits the chores with his 2 siblings who are just as fast as him.  How long, in hours, does he work?

**Reference answer:** 4
**Solution tokens:** 203 (+1 EOS supervised); full sequence 753

**Selected model solution**
```text
The bedrooms each take 20 minutes to clean, so cleaning all 3 bedrooms takes 20*3=60 minutes.
The living room takes as long as the 3 bedrooms combined, so it takes 60 minutes to clean.
The bathroom takes twice as long as the living room, so it takes 60*2=120 minutes to clean.
Cleaning the outside takes twice as long as cleaning the house, so it takes 60+60+120=240 minutes.
In total, James has to clean 60+60+120+240=480 minutes.
James splits the chores with his 2 siblings, so he only has to clean half of the work, which is 480/2=240 minutes.
There are 60 minutes in an hour, so 240 minutes is 240/60=4 hours.
#### 4
```

**Human review:** sound / substantive error / uncertain. Notes:

## 4. Shuffled index 578 — 1/4; candidate 1

**Question**
Billy ate 20 apples this week. On Monday, he ate 2 apples. On Tuesday, he ate twice as many as he ate the day before. He’s not sure what he ate on Wednesday. On Thursday, he ate four times as many as he ate on Friday. On Friday, he ate half of the amount he ate on Monday.  How many apples did he eat on Wednesday?

**Reference answer:** 9
**Solution tokens:** 70 (+1 EOS supervised); full sequence 593

**Selected model solution**
```text
Billy ate 2 apples on Monday and 2*2 = 4 apples on Tuesday.
On Friday, he ate 2/2 = 1 apple.
He ate 4*1 = 4 apples on Thursday.
He ate 20-2-4-4-1 = 9 apples on Wednesday.
#### 9
```

**Human review:** sound / substantive error / uncertain. Notes:

## 5. Shuffled index 1071 — 1/4; candidate 1

**Question**
A club is going to get additional members so that they will have 5 more than twice their current number of their members. If the club has 10 members now, how many additional members do they need?

**Reference answer:** 15
**Solution tokens:** 48 (+1 EOS supervised); full sequence 532

**Selected model solution**
```text
The club needs to have 2*10 + 5 = 25 members.
They currently have 10 members, so they need 25-10 = 15 additional members.
#### 15
```

**Human review:** sound / substantive error / uncertain. Notes:

## 6. Shuffled index 1522 — 1/4; candidate 4

**Question**
Jimmy is setting up a garden. He has three plots: one plot for green beans, one plot for carrots, and one plot for corn. Each corn plant produces 5 times as many servings of veggies as each carrot, and each green bean plant produces half as many servings as each corn plant. If each carrot produces 4 servings of vegetables, and each plot has 9 plants, how many servings of vegetables is Jimmy growing?

**Reference answer:** 306
**Solution tokens:** 76 (+1 EOS supervised); full sequence 604

**Selected model solution**
```text
Each corn plant produces 4*5=20 servings of vegetables.
Each green bean plant produces 20/2=10 servings of vegetables.
So, each plot produces 4+20+10=34 servings of vegetables.
Therefore, Jimmy is growing 34*9=306 servings of vegetables.
#### 306
```

**Human review:** sound / substantive error / uncertain. Notes:

## 7. Shuffled index 1399 — 1/4; candidate 4

**Question**
Robie filled some plastic boxes with hockey cards. There were 10 cards in each box but 5 cards were not placed in a box. He gave away 2 boxes to his friends and he has 5 boxes with him. How many cards did Robie have in the beginning?

**Reference answer:** 75
**Solution tokens:** 87 (+1 EOS supervised); full sequence 587

**Selected model solution**
```text
If Robie gave away 2 boxes to his friends, he had 5 + 2 = 7 boxes in total.
Since there were 10 cards in each box, he had 7 * 10 = 70 cards in total.
But 5 cards were not placed in a box, so he had 70 + 5 = 75 cards in the beginning.
#### 75
```

**Human review:** sound / substantive error / uncertain. Notes:

## 8. Shuffled index 613 — 1/4; candidate 4

**Question**
Tom cannot afford a normal doctor, so he goes to a discount clinic that is 70% cheaper.   It takes two visits, though, instead of 1.   A normal doctor charges $200 for a visit.   How much money does he save?

**Reference answer:** 80
**Solution tokens:** 79 (+1 EOS supervised); full sequence 576

**Selected model solution**
```text
The discount clinic charges 200 * 70 / 100 = $140 per visit.
So he pays 140 * 2 = $280.
A normal doctor charges 200 * 1 = $200.
He saves 280 - 200 = $80.
#### 80
```

**Human review:** sound / substantive error / uncertain. Notes:

## 9. Shuffled index 924 — 1/4; candidate 2

**Question**
Two apartment roommates split the rent, utilities, and grocery payments equally each month. The rent for the whole apartment is $1100 and utilities are $114. If one roommate pays $757 in all, how many dollars are groceries for the whole apartment?

**Reference answer:** 300
**Solution tokens:** 88 (+1 EOS supervised); full sequence 587

**Selected model solution**
```text
The total cost for the whole apartment is 1100 + 114 = 1214 dollars.
If one roommate pays 757 dollars, then the total cost for the apartment is 757 * 2 = 1514 dollars.
Therefore, the groceries cost 1514 - 1214 = 300 dollars.
#### 300
```

**Human review:** sound / substantive error / uncertain. Notes:

## 10. Shuffled index 901 — 1/4; candidate 2

**Question**
On Monday Buddy has 30 baseball cards. On Tuesday Buddy loses half of them. On Wednesday Buddy buys 12 baseball cards. On Thursday he buys a third of what he had on Tuesday. How many baseball cards does he have on Thursday?

**Reference answer:** 32
**Solution tokens:** 65 (+1 EOS supervised); full sequence 557

**Selected model solution**
```text
On Tuesday Buddy has 30/2=15 baseball cards.
On Wednesday he has 15+12=27 baseball cards.
On Thursday he buys 15/3=5 baseball cards.
He has 27+5=32 baseball cards on Thursday.
#### 32
```

**Human review:** sound / substantive error / uncertain. Notes:

## 11. Shuffled index 974 — 1/4; candidate 1

**Question**
Ivan has a piggy bank that can hold 100 pennies and 50 dimes. How much, in dollars, does Ivan have if he has filled his two piggy banks with those coins?

**Reference answer:** 12
**Solution tokens:** 124 (+1 EOS supervised); full sequence 609

**Selected model solution**
```text
Each piggy bank can hold 100 pennies, so Ivan has 100 pennies x $0.01 = $1 in each piggy bank.
Each piggy bank can also hold 50 dimes, so Ivan has 50 dimes x $0.10 = $5 in each piggy bank.
Therefore, Ivan has a total of $1 + $5 = $6 in each piggy bank.
Since Ivan has two piggy banks, he has a total of $6 x 2 = $12 in total.
#### 12
```

**Human review:** sound / substantive error / uncertain. Notes:

## 12. Shuffled index 900 — 1/4; candidate 3

**Question**
Martha is knitting winter clothes for her 3 grandchildren, who are all the same size (identical triplets.) Each set of clothes consists of a woolen hat, a scarf, a sweater, a pair of mittens, and a pair of wooly socks. She can knit a hat in 2 hours. A scarf takes her 3 hours. Each mitten takes her about an hour to knit. A sock takes an hour and a half. Each sweater takes her 6 hours. How long will it take her to knit all the outfits?

**Reference answer:** 48
**Solution tokens:** 167 (+1 EOS supervised); full sequence 720

**Selected model solution**
```text
She will knit 3 hats, 3 scarves, 6 mittens, 6 socks, and 3 sweaters.
So she will knit 3 hats * 2 hours/hat = 6 hours knitting hats.
She will knit 3 scarves * 3 hours/scarf = 9 hours knitting scarves.
She will knit 6 mittens * 1 hour/mitten = 6 hours knitting mittens.
She will knit 6 socks * 1.5 hours/sock = 9 hours knitting socks.
She will knit 3 sweaters * 6 hours/sweater = 18 hours knitting sweaters.
She will knit 6 + 9 + 6 + 9 + 18 = 48 hours knitting all the outfits.
#### 48
```

**Human review:** sound / substantive error / uncertain. Notes:

## 13. Shuffled index 717 — 1/4; candidate 2

**Question**
Larry spends half an hour twice a day walking and playing with his dog. He also spends a fifth of an hour every day feeding his dog. How many minutes does Larry spend on his dog each day?

**Reference answer:** 72
**Solution tokens:** 103 (+1 EOS supervised); full sequence 585

**Selected model solution**
```text
Larry spends 0.5 * 2 = 1 hour walking and playing with his dog each day.
He also spends 1 / 5 = 0.2 hour feeding his dog each day.
In total, Larry spends 1 + 0.2 = 1.2 hours on his dog each day.
Since there are 60 minutes in an hour, Larry spends 1.2 * 60 = 72 minutes on his dog each day.
#### 72
```

**Human review:** sound / substantive error / uncertain. Notes:

## 14. Shuffled index 1499 — 1/4; candidate 3

**Question**
In Hawaii, they normally get an average of 2 inches of rain a day. With 100 days left in the year, they've gotten 430 inches of rain. How many inches on average do they need to finish the year with the normal average?

**Reference answer:** 3
**Solution tokens:** 112 (+1 EOS supervised); full sequence 609

**Selected model solution**
```text
The total amount of rain needed for the year is 2 * 365 = 730 inches.
They've already gotten 430 inches of rain, so they need 730 - 430 = 300 inches of rain for the remaining 100 days.
To find the average amount of rain they need per day, we divide the total amount of rain needed by the number of remaining days: 300 / 100 = 3 inches per day.
#### 3
```

**Human review:** sound / substantive error / uncertain. Notes:

## 15. Shuffled index 807 — 1/4; candidate 3

**Question**
Holly loves to drink chocolate milk.  With breakfast, she drinks 8 ounces of chocolate milk before leaving for work.  During her lunch break, she buys a new 64-ounce container of chocolate milk and then drinks 8 ounces of it.  With dinner, she drinks another 8 ounces of chocolate milk.  If she ends the day with 56 ounces of chocolate milk, how many ounces of chocolate milk did she begin the day with?

**Reference answer:** 16
**Solution tokens:** 96 (+1 EOS supervised); full sequence 631

**Selected model solution**
```text
Holly drinks a total of 8+8=16 ounces of chocolate milk during the day.
Holly has 56+16=72 ounces of chocolate milk at the end of the day.
Holly has 64-8=56 ounces of chocolate milk remaining in the 64-ounce container.
Holly had 72-56=16 ounces of chocolate milk at the beginning of the day.
#### 16
```

**Human review:** sound / substantive error / uncertain. Notes:

## 16. Shuffled index 899 — 3/4; candidate 2

**Question**
The number of goals scored in a game against Barca by exactly two players last season accounts for 20% of all goals scored in the league. If the players scored an equal number of goals, and the total number of goals scored in the league against Barca that season is 300, calculate the number of goals each of the two players scored.

**Reference answer:** 30
**Solution tokens:** 41 (+1 EOS supervised); full sequence 556

**Selected model solution**
```text
The two players scored 20/100 * 300 = 60 goals.
Each player scored 60/2 = 30 goals.
#### 30
```

**Human review:** sound / substantive error / uncertain. Notes:

## 17. Shuffled index 982 — 2/4; candidate 2

**Question**
Jame's buys 100 head of cattle for $40,000.  It cost 20% more than that to feed them.  They each weigh 1000 pounds and sell for $2 per pound.  How much profit did he make?

**Reference answer:** 112000
**Solution tokens:** 152 (+1 EOS supervised); full sequence 652

**Selected model solution**
```text
The cost to feed the cattle is 40000 * 1.2 = $48,000.
The total cost to buy and feed the cattle is 40000 + 48000 = $88,000.
The total weight of the cattle is 100 * 1000 = 100,000 pounds.
The total revenue from selling the cattle is 100000 * 2 = $200,000.
The profit made is 200000 - 88000 = $112,000.
#### 112000
```

**Human review:** sound / substantive error / uncertain. Notes:

## 18. Shuffled index 833 — 3/4; candidate 1

**Question**
In the garden, the gecko eats 12 bugs.  The lizard eats half as many bugs as the gecko.  The frog eats 3 times as many bugs as the lizard.  The toad eats 50% more bugs than the frog. How many bugs are eaten in total by all of them?

**Reference answer:** 63
**Solution tokens:** 72 (+1 EOS supervised); full sequence 580

**Selected model solution**
```text
The lizard eats 12/2 = 6 bugs.
The frog eats 6*3 = 18 bugs.
The toad eats 18 + 18/2 = 27 bugs.
All together, they eat 12+6+18+27 = 63 bugs.
#### 63
```

**Human review:** sound / substantive error / uncertain. Notes:

## 19. Shuffled index 922 — 2/4; candidate 3

**Question**
Twice Betty's age is the cost of a pack of nuts. Doug, Betty's friend, is 40 years old. If the sum of their ages is 90, and Betty wants to buy 20 packs of the nuts, calculate the amount of money she'll pay for the packs of nuts.

**Reference answer:** 2000
**Solution tokens:** 177 (+1 EOS supervised); full sequence 682

**Selected model solution**
```text
Let's denote Betty's age as B. According to the information given, twice Betty's age is the cost of a pack of nuts, so the cost of a pack of nuts is 2B.
Doug's age is 40 years old.
The sum of their ages is 90, so we can write the equation: B + 40 = 90.
Solving for B, we get: B = 50.
Now, we know that the cost of a pack of nuts is 2B, so the cost of a pack of nuts is 2 * 50 = 100.
If Betty wants to buy 20 packs of nuts, the amount of money she'll pay for the packs of nuts is 100 * 20 = 2000.
#### 2000
```

**Human review:** sound / substantive error / uncertain. Notes:

## 20. Shuffled index 879 — 3/4; candidate 2

**Question**
There are 11 boys and 13 girls in a class. If 1 boy is added to the class, what percentage of the class are girls?

**Reference answer:** 52
**Solution tokens:** 93 (+1 EOS supervised); full sequence 567

**Selected model solution**
```text
Initially, there are 11 boys + 13 girls = 24 students in the class.
If 1 boy is added, there will be 11 boys + 1 girl = 12 boys and 13 girls = 25 students in the class.
The percentage of girls in the class is (13 girls / 25 students) * 100 = 52%.
#### 52
```

**Human review:** sound / substantive error / uncertain. Notes:

