import torch
import numpy as np

# Найпростішим способом створення тензора є виклик torch.empty().

x = torch.empty(3, 4)
print(type(x))
print(x)


# Найчастіше ви захочете ініціалізувати свій тензор певними значенням. Типовими випадками є всі нулі, усі одиниці або випадкові значення:


zeros = torch.zeros(2, 3)
print(zeros)

ones = torch.ones(2, 3)
print(ones)

torch.manual_seed(1729)
random = torch.rand(2, 3)
print(random)

# Для створення тензору, який має таку ж саму кількість вимірів і кількість комірок, як і вже існуюючий тензор, існують методи torch.*_like():

x = torch.empty(2, 2, 3)
print(x.shape)
print(x)

empty_like_x = torch.empty_like(x)
print(empty_like_x.shape)
print(empty_like_x)

zeros_like_x = torch.zeros_like(x)
print(zeros_like_x.shape)
print(zeros_like_x)

ones_like_x = torch.ones_like(x)
print(ones_like_x.shape)
print(ones_like_x)

rand_like_x = torch.rand_like(x)
print(rand_like_x.shape)
print(rand_like_x)

# Властивість .shape містить список розмірів кожного виміру тензора — у нашому випадку x є тривимірним тензором форми 2 x 2 x 3.

# Також тензори можна створювати безпосередньо з даних. Тип даних визначається автоматично.

some_constants = torch.tensor([[3.1415926, 2.71828], [1.61803, 0.0072897]])
print(some_constants)

some_integers = torch.tensor((2, 3, 5, 7, 11, 13, 17, 19))
print(some_integers)

more_integers = torch.tensor(((2, 4, 6), [3, 6, 9]))
print(more_integers)

# torch.tensor() створює копію даних.

# Встановити тип даних тензора можна кількома способами:

a = torch.ones((2, 3), dtype=torch.int16)
print(a)

b = torch.rand((2, 3), dtype=torch.float64) * 20.
print(b)

c = b.to(torch.int32)
print(c)

# Найпростіший спосіб встановити тип даних тензора — за допомогою необов’язкового аргументу dtype під час створення.

# Іншим способом встановлення типу даних є метод .to(). У комірці вище ми створюємо випадковий тензор b з плаваючою комою звичайним способом. Після цього ми створюємо c, перетворюючи b на 32-розрядне ціле число за допомогою методу .to().

# Доступні типи даних PyTorch включають: torch.bool, torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64, torch.half, torch.float, torch.double, torch.bfloat.

tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# Індексування, нарізка, об’єднання та трансляція тензорів
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)

# Об’єднання тензорів

# - torch.cat об’єднує послідовність тензорів уздовж виміру, що вже існує:
x = torch.tensor([[1, 2], [3, 4]])
y = torch.tensor([[5, 6]])
# Конкатенація вздовж 0-го виміру
result = torch.cat((x, y), dim=0)
print(result)

# - torch.stack об’єднує послідовність тензорів у новому вимірі, створюючи новий вимір в отриманому тензорі.

x = torch.tensor([[1, 2], [3, 4]])
y = torch.tensor([[5, 6], [3, 4]])
# Стек вздовж нового виміру (вимір 0)
result = torch.stack((x, y), dim=0)
print(result)

# Трансляція тензорів (tensor broadcasting)
rand = torch.rand(2, 4)
doubled = rand * (torch.ones(1, 4) * 2)

print(rand)
print(doubled)

'''
Трансляція — це спосіб виконання операції між тензорами, які мають подібні форми. У наведеному вище прикладі тензор з одним рядком і чотирма стовпцями множиться на обидва рядки тензора з двома рядками і чотирма стовпцями.

Це важлива операція в Deep Learning. Звичайним прикладом є множення тензора вагових коефіцієнтів навчання на групу вхідних тензорів, застосування операції до кожного екземпляра в групі окремо та повернення тензора ідентичної форми. Так, у наведеному вище прикладі множення тензорів розмірами (2, 4) та (1, 4) було повернуто тензор форми (2, 4).

Правила трансляції:
- Кожен тензор повинен мати принаймні один вимір — ніяких порожніх тензорів.
- Порівняння розмірів двох тензорів від останнього до першого:
розміри вимірів мають бути рівними
або
один із вимірів має розмір 1
або
розмірність не існує в одному з тензорів.
'''

a = torch.ones(4, 3, 2)

b = a * torch.rand(   3, 2) # 3rd & 2nd dims identical to a, dim 1 absent
print(b)

c = a * torch.rand(   3, 1) # 3rd dim = 1, 2nd dim identical to a
print(c)

d = a * torch.rand(   1, 2) # 3rd dim identical to a, 2nd dim = 1
print(d)

# Одноелементні тензори
# Якщо у вас є одноелементний тензор, отриманий, наприклад, шляхом об’єднання всіх значень тензора в одне значення, ви можете перетворити його на числове значення Python за допомогою методу item().

agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

# Локальні операції (In-place operations)
# Операції, які зберігають результат в операнді, є локальними. Вони позначаються суфіксом _. Наприклад: *x.copy*(y), x.t_() змінить x.
print(f"{tensor} \\n")
tensor.add_(5)
print(tensor)

# Тензори на CPU і масиви NumPy можуть спільно використовувати пам’ять, і зміна одного з масивів змінить інший.
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# Приведення масиву NumPy до tensor
n = np.ones(5)
t = torch.from_numpy(n)

# Зміни в масиві NumPy відображаються в тензорі.

np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")

# Переносимо наш тензор на графічний процесор (GPU), якщо він доступний
if torch.cuda.is_available():
    tensor = tensor.to("cuda")

    # Робота з розмірністю тензорів
# Отже, як зробити батч з одного зображення?

a = torch.rand(3, 226, 226)
b = a.unsqueeze(0)

print(a.shape)
print(b.shape)

# Виведення:

# torch.Size([3, 226, 226])
# torch.Size([1, 3, 226, 226])

# Метод unsqueeze() додає вимір розміру 1. unsqueeze(0) додає його як новий нульовий вимір — тепер ми маємо батч розміру 1.

# Що робити у випадку, коли вам знадобиться зробити обчислення з самим результатом — 20-елементним вектором?

a = torch.rand(1, 20)
print(a.shape)
print(a)

b = a.squeeze(0)
print(b.shape)
print(b)

c = torch.rand(2, 2)
print(c.shape)

d = c.squeeze(0)
print(d.shape)

# 💡 Виклики squeeze() і unsqueeze() можуть діяти лише на вимірах розмірності 1, оскільки, інакше, зміниться кількість елементів у тензорі.


# Одновимірний результат можна отримати за допомогою операції reshape() за умови, що виміри, які ви хочете отримати, дають таку саму кількість елементів, яку має вхідний тензор:

# 💡 Аргумент (6 * 20 * 20,) у останньому рядку комірки вище пояснюється тим, що PyTorch очікує кортеж, коли вказує форму тензора, але коли форма є першим аргументом методу. Це дозволяє нам «шахраювати» та з легкістю використовувати ряд цілих чисел. 
# Тут нам довелося додати дужки та кому, щоб переконати метод, що це дійсно одноелементний кортеж.

output3d = torch.rand(6, 20, 20)
print(output3d.shape)

input1d = output3d.reshape(6 * 20 * 20)
print(input1d.shape)

# також може викликати його як метод у модулі torch:
print(torch.reshape(output3d, (6 * 20 * 20,)).shape)

# ЗАДАЧА

from torch import nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

m = nn.Linear(5, 3)
input = torch.randn(4, 5)
output = m(input)

print('Input:', input, f'shape {input.shape}', sep='\\n')
print('\\nOutput:', output, f'shape {output.shape}', sep='\\n')

'''
Сигмоіда — це наша функція активації, реалізована в PyTorch як torch.sigmoid.

torch.sigmoid(*input*, *, *out=None)

Параметри

input (Tensor) – вхідний тензор;
out (Tensor, необов’язково) – вихідний тензор. Вихідним тензором може бути заздалегідь створений об’єкт torch.tensor, в який буде збережено результат розрахунку.
'''

# Визначимо клас логістичної регресії.

class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear(x)
        out = self.sigmoid(out)
        return out

df = pd.read_csv('data/train.csv')
df = df.set_index('PassengerId')

TARGET = 'Transported'
FEATURES = [col for col in df.columns if col != TARGET]

text_features = ["Cabin", "Name"]
cat_features = [col for col in FEATURES if df[col].nunique() < 25 and col not in text_features ]
cont_features = [col for col in FEATURES if df[col].nunique() >= 25 and col not in text_features ]

ax = df[TARGET].value_counts().plot(kind='bar', figsize=(8, 5))
for i in ax.containers:
  ax.bar_label(i)
  ax.set_xlabel("value")
  ax.set_ylabel("count")
       
plt.suptitle("Target feature distribution")

plt.tight_layout()
plt.show()


ax = df.loc[:, cont_features].hist(figsize=(10, 12), grid=False, edgecolor='black', linewidth=.4)
for row in ax:
  for col in row:
    for i in col.containers:
      col.bar_label(i)
      col.set_xlabel("value")
      col.set_ylabel("count")
     
services_features = cont_features[1:]

for feature in services_features:
    df[f'used_{feature}'] = df.loc[:, feature].apply(lambda x: 1 if x > 0 else 0)

# Correlation matrix for selected features
corr_matrix = df.loc[:, cont_features + ['CryoSleep', 'VIP', TARGET]].corr()

# Display a styled correlation matrix (optional, requires jinja2)
try:
    print(corr_matrix)
except:
    print("Install 'jinja2' to use .style on DataFrames")

imputer_cols = ["Age", "FoodCourt", "ShoppingMall", "Spa", "VRDeck" ,"RoomService"]
imputer = SimpleImputer(strategy='median')
imputer.fit(df[imputer_cols])
df[imputer_cols] = imputer.transform(df[imputer_cols])
df["HomePlanet"].fillna('Gallifrey', inplace=True)
df["Destination"].fillna('Skaro', inplace=True)

df['CryoSleep_is_missing'] = df['CryoSleep'].isna().astype(int)
df['VIP_is_missing'] = df['VIP'].isna().astype(int)

df["CryoSleep"].fillna(False, inplace=True)
df["VIP"].fillna(False, inplace=True)

df["CryoSleep"] = df["CryoSleep"].astype(int)
df["VIP"] = df["VIP"].astype(int)

dummies = pd.get_dummies(df.loc[:, ['HomePlanet', 'Destination']], dtype=int)
dummies

df = pd.concat([df, dummies], axis=1)
df.drop(columns=['HomePlanet', 'Destination'], inplace=True)

# Оскільки модель, яку ми будемо створювати, очікує на числові значення, перетворюємо цільову ознаку з бінарної на цілочисельну.

df[TARGET] = df[TARGET].astype(int)

# Оскільки наразі ми не обробляємо текстові змінні, видалимо їх.

df.drop(["Name" ,"Cabin"] , axis=1 ,inplace = True)

# Train/test split

X = df.drop(TARGET , axis =1 )
y = df[TARGET]

X_train , X_test , y_train , y_test = train_test_split(X, y, random_state = 42, test_size =0.33, stratify=y)

input_dim = X_train.shape[1]
model = LogisticRegression(input_dim)

# Визначаємо функцію втрат. Модуль nn містить в собі реалізовані функції втрат, в тому числі binary cross-entropy.

criterion = nn.BCELoss()

optimizer = optim.SGD(model.parameters(), lr=0.01)



num_epochs = 50
for epoch in range(num_epochs):
    # Передача вперед
    outputs = model(X_train)
    loss = criterion(outputs.squeeze(), y_train)
    
    # Зворотний прохід та оптимізація
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

with torch.no_grad():
    y_pred = model(X_test).squeeze().numpy().round()

accuracy_score(y_test, y_pred)