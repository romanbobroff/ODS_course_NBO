# **Проект ML NBO**
## **1 Цели и предпосылки**
### **1.1 Обоснованность разработки продукта**

#### **Бизнес цель**

Разработать систему автоматизированного расчета Next Best Offer (далее персональное предложение) с целью получить дополнительный инкремент маржи относительно baseline.
Также автоматизированный расчет позволит сократить затраты на «ручной труд» в коммерческом и маркетинговых блоках (создание и согласование промоактивностей).

#### **Предпосылки**

Компания – лидер на российском рынке в категории товаров DIY более 20 лет с высокой долей продаж в оффлайн через собственную сеть гипермаркетов (более 100 магазинов) и низкой долей продаж в онлайн (интернет-магазин).

Компания столкнулась с тем, что бОльшая часть массовых снижений регулярной цены неэффективна. В условиях усиливающейся конкуренции прямое снижение цен на товары относительно рынка и поддержка стратегии Every Day Low Price (далее EDLP) стратегии ценообразования является неэффективной как с точки зрения P&L компании, так и с точки зрения маркетинговых целей привлечения и удержания клиентов.

Основные причины неэффективности массовых снижений регулярных цен, распродаж, промоактивностей, которые приводят к прямым потерям в марже:
- неоднородная эластичность товаров и категорий товаров;
- неоднородная сезонность спроса;
- различная чувствительность к скидкам клиентских сегментов;
- высокий уровень каннибализации категорий в период снижения цен или распродаж;
- высокий уровень хало-эффекта: снижение спроса после возврата цены на прежний уровень при промоактивностях;
- потери от неточного прогнозирования количества товаров в момент проведения промо, что ведет к дефициту (снижению эффективности) или к затовариванию остатков;
системная реакция конкурентов на изменения цен в магазинах компании.

Другими словами, цель проекта:

Декомпозировать сущности товар – цена – канал коммуникации по отношению у сущности клиент до уровня, когда декомпозиция будет приносить инкрементальную выгоду (выручку или маржу).

#### **Польза от внедрения проекта**

В компании было принято решение разработать систему автоматизированного расчета персональных предложений на клиента в личный канал коммуникации.

Под параметрами персонального предложения считаются:
- клиент(ы)
- дата следующей покупки
- товары или группы товаров 
- тип механики промо
- тип поощрения
- значения промо-механик (условия выполнения, глубина поощрения)
- периоды выполнения условий промо-механик и использования поощрения
- оптимальный(ые) канал(ы) коммуникации
- оптимальный период для коммуникации.

Результатом проекта должен быть расчет всех параметров персонального предложения на клиента, при этом типы поощрений должны быть в % от регулярной цены, действующий на период применения поощрения.

Необходимые данные для реализации проекта:
- транзакции (продажи и возвраты)
- выгрузки из GA или YM по поведению на сайте и в мобильном приложении
- справочник товаров и категорий (иерархия)
- справочник клиентов
- справочник атрибутов товаров
- справочник каналов продаж и магазинов
- справочник механик промо 
- справочник типов цен
- справочник каналов коммуникаций
- история коммуникаций с клиентами в персональных каналах.


Риски проекта:
- Инкремент маржи будет ниже планового;
- Клиенты привыкли к низким регулярным ценам и какое-либо поощрение в персональном канале может оказаться незамеченным;
- Сложный проект с точки зрения изменений в ИТ архитектуре, бизнес-процессах, может быть не доведен до финальной точки.

Архитектура решения персонального предложения состоит из трех последовательных MVP проектов, где результат предшествующего MVP является основой для следующего MVP, при этом часть параметров вычисляются на уровне ML, а часть на основе бизнес-правил (ссылка на схему в директории).



Длительность всего проекта оценивается на уровне 2 календарных лет с учетом необходимого времени на проведение A/B тестов и подведения итогов для перехода в следующий MVP.

В этом проекте мы рассматриваем только MVP1 (далее MVP), в результате которого мы сможем:
- ML-модели: предсказать дату(ы) следующей покупки клиентом;
- ML-модели: предсказать товар(ы) и/или категорию(ии) товаров следующей покупки;
- Бизнес-правила: рассчитать вариант оптимальной механики предложения, задать канал коммуникации и период действия персонального предложения.

#### **Критерии успеха MVP**
##### **Бизнес-метрики**
Получение инкрементальной выручки на уровне +5%, очищенной от суммы предоставленных скидок в виде поощрений на персональное предложение.
Величина инкрементальной выручки задается экспертно, с учетом того, что на данном этапе развития компания теряет до 3% выручки после проведения распродаж и других промоактивностей.

##### **Метрики ML**
- Модели, предсказывающие следующую дату покупки, оцениваются по средней абсолютной ошибке (MAE).  В результате расчета выбирается модель с самым низким значением метрики. Пороговое значение принятия результатов модели MAE<=14 дней.

- Метрики модели предсказания товара и категории товаров:
  - recall@[1, 5, 10, 20] - “полнота” - отношение верно предсказанных из топ-[1, 5, 10, 20] ко всем купленным на тестовом периоде (измеренное по каждому юзеру и затем усредненное)
  - precision@[1, 5, 10, 20] - “точность” - отношение верно предсказанных из топ-[1, 5, 10, 20] ко всем предсказанным из топ - [1, 5, 10, 20] на тестовом периоде (измеренное по каждому юзеру и затем усредненное)
  - users_hitted@[1, 5, 10, 20] - доля юзеров, для которых был правильно рекомендован хотя бы один товар из топ-[1, 5, 10, 20]

В результате расчета моделей предсказания выбирается модель с самым высоким значением метрики.

Пороговое значение принятия результатов модели >= 25%.

- Метрика по определению оптимальной механики по результатам проведения AB тестов

Пороговое значение для принятия решения – инкремент выручки +5% (за минусом скидок) в контрольной группе относительно тестовой.

### 1.2 Бизнес-требования и ограничения

Данные представлены:

- за три календарных года или 36 месяцев;
- по одному из магазинов сети;
- гранулярность данных – линейная чековая запись;
- общий объем – 4,5 млн линейных записей;
- всего 410 000 клиентов, в среднем по 11 линейных записей на 1го клиента;
- всего 200 000 SKU.

Ограничения, допущения для достижения критериев успеха MVP:
- отсутствуют данные по региону и иным характеристикам расположения магазина;
нет данных по остаткам товаров, необходимые при расчете некоторых агрегированных показателей витрины и для скоринга предсказаний;
- отсутствуют внешние данные, такие как погодные условия, курсы валют, доходы населения региона, календарь праздников, периоды карантина и другие;
- отсутствуют данные по поведению клиента на сайте до или во время покупки, в том числе данные по поисковым запросам;
- транзакционные данные не включают товары, которые в указанный период ни разу не продавались;
- товары имеют свой цикл обновления, в указанной выборке могут быть новинки, на которых модель не обучалась;
- также в выборке могут быть новые клиенты, у которых не было транзакций в тестовой выборке;
- в компании нет исторических данных по проведению промо в различных каналах коммуникации;
- отсутствуют данные по марже, поэтому при выборе оптимальной механики выбирается максимум по инкременту выручки;
- выбран только один уровня иерархии товаров (category);
- компания пока не использует бонусы как отдельный тип поощрения в промо механиках;
- компания не планирует обогащать 3rd party данными;
- отсутствуют данные по оценкам и отзывам товаров.

В целях обучения моделей предсказания планируем часть данных использовать в качестве тренировочной выборки, оставшуюся по календарному периоду часть использовать в качестве тестовой выборки в следующих пропорциях:
- 34 месяца и 2 месяца
- 30 месяцев и 6 месяцев.

Гранулярность предсказания:
- Клиент
- Товар(ы) и/или категория(ии) 
- Дата следующей покупки.

## **2. Методология MVP**
### **2.1 Постановка задачи**
#### 2.1.1 Решение задачи Предсказания товара(ов) и категории(ий) планируется реализовать путем построения следующих моделей:
- кластеризация клиентских данных на k-means;
- кластеризация продуктовых корзин на k-means;
- модель ассоциативных правил;
- модель коллаборативной фильтрации;
- модель цепи Маркова;
- модели градиентного бустинга и ансамблирование моделей с постобработкой результатов.

#### 2.1.2 Решение задачи Предсказания даты следующей покупки планируется реализовать путем построения:
- моделей класса BTYD (Buy Till You Die), которые моделируют процесс совершения повторных покупок, пока клиенты еще «живы»;
- расчета и аналитических «манипуляций» с median duration в матрице переходов из модели Цепи Маркова.

#### 2.1.3. Решение задачи Выбора оптимальной механики планируется реализовать на основе результатов этапа 2.1.1 и модели ассоциативных правил путем перебора комбинаторик товаров из связи XY и значений Lift.

### **2.2 Блок-схема MVP (ссылка на репозиторий)**

#### **Этап 1. Подготовка baseline среды**
##### **Этап 1.1. Сбор данных**

Скачиваем данные о транзакциях из системы. Загружаем их в тетрадь юпитера. При отборе проверяем заполненность данных – убираем малозаполненные данные или малодостоверные. 

Данные (ссылка), которые компания смогла предоставить для MVP:

| Дано              | Описание |
| :---------------- | :------ |
|   id     |  id клиента    |
|   purch_date     |    Дата чека  |
|   channel     |   Канал продаж (ONLINE, OFFLINE)   |
|   category_name     |    Категории товара, список  |
|   category     |  Категория товара, третий уровень детализации    |
|   description     |   Название товара   |
|  item      |  Уникальный артикул товара    |
|   price     |  Цена в чеке    |
|    quantity    |  Количество в чеке    |
|   turnover     |  Сумма по строке    |
|    uom    |  Единица измерения товара (EA = штуки)    |
|    packing_size    |  Размер упаковки товара    |

##### **Этап 1.2. Обработка данных**

Не чистим от выбросов, так как непонятно что является выбросом (специфика «разброса» данных для категории DIY;

Объединяем чеки с одной датой на клиента в один уникальный номер чека;

Рассчитываем агрегированные показатели витрины.

| Название              | Описание |
| :---------------- | :------ |
|   id_check_unic     |  Номер чека    |
|is_purchase | Флаг покупки: 1 = Покупка, 0 = Возврат|
| price_uom| Цена за единицу товара|
|price_% | % изменения цены|
| price_cat1| Фича 1, основанная на анализе % изменения цены|
| price_cat2| Фича 2, основанная на анализе категорий товара|
| purch_date_year|Год покупки |
| purch_date_mounth| Месяц покупки|
| purch_date_week| Номер календарной недели покупки|
| purch_date_day| Номер дня недели покупки 1=ПН, 2=ВТ… 7=ВС|
| purch_date_day type| Будний или выходной - день покупки|
| purch_quantity_category_year| Кол-во проданных штук в категории в текущем году|
| purch_quantity_category_week| Кол-во проданных штук в категории на текущей неделе в текущем году|
| purch_season_share| Сезонность, доля проданных штук в категории за неделю к году|
| purch_season type| Фича - квартили по сезонности (значения 1, 2, 3, 4)|
|price_type | Фича - квартили по альтернативной цене (значения 1, 2, 3, 4)|
|elasticity_category_week |Фича - эластичность цены |
|ltv_turnover | ТО по клиенту за весь период|
|ltv_quantity |Кол-во купленных штук по клиенту за весь период |
| ltv_check_count| Кол-во чеков с ПОКУПКАМИ по клиенту за весь период|
|ltv_purch_date_count |Кол-во дней с ПОКУПКАМИ по клиенту за весь период |
| ltv_item_count| Кол-во уникальных КУПЛЕННЫХ товаров по клиенту за весь период|
| check_av_turn| Средний чек по клиенту|
|check_av_quintity | Среднее кол-во штук товара в чеке по клиенту|
| check_av_item| Среднее кол-во уникальных артикулов в чеке по клиенту|
|frequence_client_per_month |Среднее кол-во дней с покупками в месяц по клиентам |
|days_between_min_max_date |Кол-во дней между первой и последней датами покупок по всему датасету |
| recency_client| Кол-во дней с момента последней даты покупки до последней даты покупок в датасете|
| ltv_quantity_cumul| Накопительное кол-во купленных штук по клиенту за весь период|
| monetary| Накопительный ТО по клиенту за весь период|
| frequency| Накопительно частота покупок по клиенту за весь период|
| recency| Кол-во дней с момента последней даты покупки до текущей даты|
| elasticity_client| Фича - эластичность клиента|

Расчет показателей витрины – раз в сутки.

##### **Этап 1.3. Обучение моделей (ссылка на схему в репозитории)**

**Модель 1 «Кластеризация клиентов»** и **модель 2 «Кластеризация продуктовых корзин»** используют k-means, результат модели как дополнительные фичи подается на последующие модели для расчета предсказаний. 

? Роман Описываем результат в виде количества полученных кластеров, расстояния между ними, выставляем картинки-графики 

**Модель 3. Ассоциативные правила** (или Apriori)

Считалась двумя способами:
- внутри одного;
- lastnext, где каждый последующий чек объединялся с предыдущим чеком, создавая новый уникальный номер чека.

**Модели 4-6. Коллаборативная фильтрация: implicit TF-IDF, BM25, Cosine**

![Alt text](https://takuti.github.io/Recommendation.jl/latest/assets/images/user-item-matrix.png)

Коллаборативная фильтрация использует данные о взаимодействиях пользователей и объектов для построения векторных представлений, чтобы на основе них найти функцию, которая будет предсказывать оценку релевантности объектов для пользователей.

Github - https://github.com/benfred/implicit 

Docs - https://implicit.readthedocs.io/en/latest/ 

Доступные модели:
- Item2item kNN
- Logistic matrix factorization
- implicit ALS
- Bayesian Personalized Ranking

Используемые в нашем проекте TF-IDF, BM25, Cosine - это item2item kNN с несколькими вариантами предобработки item2item матрицы.

![Alt text](https://takuti.github.io/Recommendation.jl/latest/assets/images/cf.png)

Основная идея item2item kNN-подхода:

1. В качестве векторных представлений для пользователей и объектов использовать строки и столбцы из матрицы взаимодействий;

2. С помощью векторов оцениваем схожесть и на основе нее строим рекомендации.

В implicit доступны три модели:
- CosineRecommender (модель с косинусным расстоянием)
- TFIDFRecommender (перевзвешивает входную матрицу методом TF-IDF)
- BM25Recommender (перевзвешивает входную матрицу методом BM25)

Гиперпараметр один для всех - К, число соседей. Он влияет на топ, который может выдать модель (лучше ставить его не менее 30). 

Обучение: считаем матрицу расстояний между объектами, но храним не всю, а K расстояний до каждого из наиболее похожих объектов.

Построение рекомендаций для пользователя:

1.   Получение топ-К соседей для каждого объекта, с которым пользователь взаимодействовал
2.   Объединение всех топов в один с суммированием схожести
3.   Выдача топ-N самых похожих объектов

**Модель 7. Цепи Маркова (кастомная реализация)**

Цепи Маркова — это последовательность событий или действий, где каждое новое событие зависит только от предыдущего и не учитывает все остальные события. Такой алгоритм не помнит, что было раньше, а смотрит только на предыдущее состояние.

Для каждой пары последовательных покупок мы считаем их количество, на их основе строим матрицу количества переходов, которые конвертируем в вероятности. Рекомендации для пользователя получаем за счет перемножения вектора его покупок на матрицу вероятностей.

**Модели 8-9. TopPopular, UserTopPopular**

Это эвристики, которые каждому пользователю выдают топ-K популярных товаров или топ-K самых покупаемых им товаров. Эти модели вносят Popularity bias в рекомендации, что не очень хорошо, но служат хорошими бэйзлайнами для оценки других, более сложных моделей, а также могут использоваться в комбинации с ними.

**Модель 10. Градиентный бустинг: CatBoost**

У простых моделей есть несколько недостатков:
- нет учёта временной составляющей
- линейность 
- признаки, ранки, эмбеддинги не используются
- неизвестно, как блендить результаты

Двухэтапная модель делает это автоматически, ранжируя рекомендации от простых моделей. Процесс выглядит так:
1. Ко всем товарам на платформе применим некоторую фильтрацию
2. Обучим одну или несколько легких моделей. Их результатом будут являться рекомендации
3. Обучим ранжирующую модель (бустинг или нейросеть) и скорим ею айтемов-кандидатов от простых моделей. Ранжируем и отбираем айтемы по этому скору, получая итоговые рекомендации для пользователей.

![Alt text](https://drive.google.com/uc?export=view&id=1s6c3RVRBMssSyGhcQHOpLhTaRXRDxaJC)

Если оставить только ранжирование бустингом на втором шаге и использовать его для проставления рангов у всех доступных айтемов для каждого пользователя, то будут затрачены большие вычислительные ресурсы, потому что количество юзер-айтем-пар будет исчисляться миллионами (количество пользователей Х количество айтемов).

**Цель модели второго этапа** - это переранжировать пары юзер-айтем, оставшиеся после первого этапа так, чтобы метрики выросли. Одна из популярных постановок задачи, которая хорошо работает - это задача бинарной классификации на парах юзер-айтем. В качестве модели используют бустинг над деревьями (XGBoost, LightGBM, CatBoost). В качестве функции потерь - Logloss. Также можно обучать ранжирование напрямую, используя, к примеру, YetiRank в CatBoost. В некоторых задачах задача может быть поставлена как задача регрессии, когда нужно восстановить какой-нибудь вещественный target, например, процент от просмотра или таргет с учетом весов разных видов отклика от пользователя.

В случае постановки задачи как бинарной классификации **обучающая выборка** будет выглядеть как матрица, где по строкам сконкатенированы все признаки по пользователю, товару, взаимные признаки и признаки от моделей первого этапа. Таргет будет =1, если пользователь взаимодействовал с товаром, и 0, если нет.

![Alt text](https://drive.google.com/uc?export=view&id=1eCj0ck7XDO1LaqDrK4UiidWaEbTjBiT5)
 
**Схема валидации** получается такая:
1. Данные разделяются по времени на глобальные train и test: на train будем обучать модели, а на test - проверять их метрики качества.
2. Выделенный train делится на 2 части: на одной обучаем модели первого этапа, на второй - предсказываем кандидатов, на ней же обучаем модель второго этапа на размеченных и обогащенных признаками кандидатах

![Alt text](https://drive.google.com/uc?export=view&id=1g6hXskxi_20rg_QtGSLjVkCL-jlwskXP)

Обучение моделей в нашем проекте строится следующим образом:

В режиме валидации (который запускается по необходимости для разработки и снятия метрик) - для каждой из схем (“предсказание товара” и “предсказание категории”):

1. Откладываем тестовую выборку (2 или 6 месяцев) и не трогаем её до конца;
2. От тренировочной снова откладываем малую тестовую (2 месяца); 
3. На малой тренировочной обучаем базовые модели (генераторы кандидатов), предсказываем кандидатов на малую тестовую
4. Объединяем кандидатов
5. На малой тестовой, смотря на кандидатов и реальные покупки, размечаем кандидатов: 1 - если была покупка, и 0 - если нет; 
6. К кандидатам приписываем признаки юзеров, айтемов, совместные юзеров-айтемов, ранги и скоры базовых моделей;
7. После обучения базовых моделей обучаем бустинг на этих кандидатах с признаками на таргет 0/1;
8. Переобучаем все базовые модели на всём трейне. Ответы сравниваем с большой тестовой выборкой, снимаем метрики;
9. Делаем предсказание: прогоняем датасет через базовые модели и бустинг.
10. Ответы сравниваем с большой тестовой выборкой, снимаем метрики.
11. Собираем предсказания схем “предсказание товара” и “предсказание категории” в один список, переранжируем по скору. Ответы сравниваем с большой тестовой выборкой, снимаем метрики.

В продовом режиме (который запускается 1 раз в сутки) - для каждой из схем (“предсказание товара” и “предсказание категории”):

1. Откладываем малую тестовую выборку (2 месяца);
2. На малой тренировочной обучаем базовые модели (генераторы кандидатов), предсказываем кандидатов на малую тестовую, объединяем кандидатов;
3. На малой тестовой, смотря на кандидатов и реальные покупки, размечаем кандидатов: 1 - если была покупка, и 0 - если нет; 
4. К кандидатам приписываем признаки юзеров, айтемов, совместные юзеров-айтемов, ранги и скоры базовых моделей;
5. После обучения базовых моделей обучаем бустинг на этих кандидатах с признаками на таргет 0/1;
6. Переобучаем все базовые модели на всём трейне;
7. Делаем предсказание: прогоняем датасет через базовые модели и бустинг.
8. Собираем предсказания схем “предсказание товара” и “предсказание категории” в один список, переранжируем по скору;
9. Отправляем рекомендации в БД Redis.

#### **Объединение моделей предсказания товара и категории**

В результате обучения моделей “предсказание товара” и “предсказание категории” у нас получаются отранжированные списки рекомендаций товаров и категорий для пользователей соответственно (со своими скорами). Далее мы объединяем эти списки и ранжируем по скору, получая итоговый ранг - merge_rnk.

#### **Предсказание даты следующей покупки**

Подсчёт медианных (по всем юзерам) временных промежутков для каждой пары товаров на 1м сплите
Предсказание дат следующих покупок по медианным временам между товарами последней корзины и рекомендациям

#### **Эксперименты**

Первая часть экспериментов проводились на комбинациях базовая модель + бустинг, для определения наиболее сильных базовых моделей (как отдельно, так и в комбинации с бустингом). Далее собирались разные комбинации базовых моделей + бустинг.

В задаче предсказания товара среди вариантов базовая модель + бустинг лучше всего себя показали модели:

- Apriori - лучшая
- Apriori lastnext
- TopPopular
- TF-IDF

Комбинации Apriori с остальными базовыми моделями не дали прироста по реколлу, поэтому в итоге лучший пайплайн - **Аpriori + boosting**.

В предсказании категории лучшими базовыми моделями оказались:
- TopPopular
- Apriori
- Apriori lastnext
- BM25

Среди комбинаций наиболее удачной получилась:
**BM25 + Markov chain + TopPopular + UserTopPopular + Apriori + boosting.**
Результаты полученных метрик в таблицах.
Важность признаков для модели предсказания товара:

Важность признаков для модели предсказания категории:

Результат расчета ML-моделей: выдаются предсказанные на уровне клиента товары (item) и категории (category) вперемешку - со значением merge_rnk (чем меньше merge_rnk, тем релевантнее, по мнению гибридной модели, товар).

**Этап 1.4. Бизнес-правила по определение механики (ссылка на файл в Ексель + ссылка на схему)**


Выбор оптимальной механики реализован на основе модели ассоциативных правил путем перебора комбинаторик товаров из связи XY для выбора максимального инкремента выручки.
Если 
Результат модели ассоциативных правил
- category_X / Item X
- category_Y / Item Y
- support
- confidence
- lift
- $P_{медиана}(X)$
- $P_{ср}(X)$
- $Q_{медиана}(X)$
- $Q_{ср}(X)$
- $P_{медиана}(Y)$
- $P_{ср}(Y)$
- $Q_{медиана}(Y)$
- $Q_{ср}(Y)$,

Где:

- $X$ и $Y$ – item или category;
- support, confidence, lift – параметры, описывающий связь между $XY$;
- $Q$ – количество, соответственно $Q(X)$ и $Q(Y)$
- $S$ – сумма, соответственно $S(X)$ и $S(Y)$
- $V$ – выручка, соответственно $V(X)$ и $V(Y)$

Медиана и средние – статистические показатели, соответственно $Q(X)$ медиана, $Q(X)$ средний, $Q(Y)$ медиана, $Q(Y)$ средний, $S(X)$ медиана, $S(X)$ средний, $S(Y)$ медиана, $S(Y)$ средний.

Типы механик:
- (Купи $Q (X)$ получи $Q (X)$ c поощрением)
- (Купи $S (X)$ получи $S (X)$ c поощрением)
- (Купи на $((Q (X) + Q (Y))$ - получи поощрение)
- (Купи на $((S (X) + S (Y))$ - получи поощрение),

то выбор оптимальной механики – максимальное значение инкрементальной выручки с 1 чека

Тип механики

| Тип механики         | Расчет инкрементальной выручки без поощрения |
| :---------------- | :------ |
| (Купи $Q(X)$ получи $Q (X)$     |   $(Q_{ср}(X) - Q_{медиана} (X))*V(X)$   |
| (Купи $S(X)$ получи $S (X)$ c поощрением), руб. | $P_{ср} (X)*Q_{ср} (X) - P_{медиана} (X)*Q_{медиана} (X)$ |
|(Купи на $((Q (X) + Q (Y))$ |$P_{медиана} (Y) * Q_{медиана} (Y)$|
|(Купи на $((S (X) + S (Y))$ |$(Q_{ср} (X) - Q_{медиана} (X))*V(X) + (Q_{ср}(Y) - Q_{медиана}(Y))*V(Y)$|

Тип поощрения:

| Эластичность товара / категории              | In Stock | Price |
| :---------------- | :------ | :---- |
| эластичный       |   Скидка, %   | Бонус начисление, % |
| неэластичный           |   Бонус начисление, %   | исключение из рекомендаций |

Эластичность товара / категории
Эластичность клиента
эластичный
неэластичный
эластичный


Ограничения, допущения:
- результаты подбора механики транспонируются на результаты полученных рекомендаций (ML моделей) на клиента, обрабатываются на удаление дублей по item и category;
- оптимальное количество персонального предложения на клиента, исходя из cumulative (merge_rnk) с кратностью 2 и максимумом в 8 предложений;
- бизнес планирует использовать на этапе MVP один канал коммуникации – личный кабинет на сайте;
- бизнес определил экспертно оптимальный период действия промомеханики – 7 календарных дней со среды по вторник;
- дополнительный скоринг на merge_rnk с учетом сезонности;
