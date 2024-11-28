# Python Script to Display an Emotions Tree Using NLTK (Enhanced)
from nltk.tree import Tree

# Define the emotion tree structure
emotions_tree_ussr = Tree(
    'Эмоции (Emotions)',
    [
        Tree('Положительные (Positive)', [
            Tree('Радость (Joy)', [
                'Счастье (Happiness)',
                'Удовлетворение (Contentment)',
                'Веселье (Merriment)',
                'Восторг (Delight)'
            ]),
            Tree('Любовь (Love)', [
                'Привязанность (Affection)',
                'Сострадание (Compassion)',
                'Восхищение (Admiration)',
                'Благодарность (Gratitude)'
            ]),
            Tree('Удивление (Surprise)', [
                'Изумление (Amazement)',
                'Восхищение (Awe)'
            ])
        ]),
        Tree('Отрицательные (Negative)', [
            Tree('Гнев (Anger)', [
                'Раздражение (Irritation)',
                'Ярость (Rage)',
                'Недовольство (Discontent)',
                'Негодование (Indignation)'
            ]),
            Tree('Печаль (Sadness)', [
                'Горе (Grief)',
                'Разочарование (Disappointment)',
                'Скука (Boredom)',
                'Одиночество (Loneliness)'
            ]),
            Tree('Страх (Fear)', [
                'Тревога (Anxiety)',
                'Беспокойство (Worry)',
                'Испуг (Fright)',
                'Паника (Panic)'
            ]),
            Tree('Отвращение (Disgust)', [
                'Презрение (Contempt)',
                'Неприязнь (Aversion)',
                'Омерзение (Loathing)'
            ])
        ]),
        Tree('Смешанные (Mixed)', [
            Tree('Ностальгия (Nostalgia)', [
                'Светлая грусть (Bittersweet Melancholy)',
                'Тоска (Longing)'
            ]),
            Tree('Амбивалентность (Ambivalence)', [
                'Сочетание радости и печали (Joy and Sadness Mix)'
            ])
        ])
    ]
)
emotions_tree_france_1940 = Tree(
    'Émotions (Эмоции)',
    [
        Tree('Positives (Положительные)', [
            Tree('Joie (Радость)', ['Liberté retrouvée (Restored Freedom)', 'Espoir (Hope)', 'Succès culturel (Cultural Success)']),
            Tree('Amour (Любовь)', ['Amour romantique (Romantic Love)', 'Fraternité (Brotherhood)', 'Solidarité (Solidarity)']),
        ]),
        Tree('Négatives (Отрицательные)', [
            Tree('Colère (Гнев)', ['Indignation (Негодование)', 'Frustration sociale (Social Frustration)']),
            Tree('Tristesse (Печаль)', ['Deuil (Mourning)', 'Perte (Loss)', 'Solitude (Loneliness)']),
            Tree('Peur (Страх)', ['Peur de l’inconnu (Fear of the Unknown)', 'Inquiétude (Worry)'])
        ]),
        Tree('Mixtes (Смешанные)', [
            Tree('Nostalgie (Ностальгия)', ['Souvenirs de guerre (Memories of War)', 'Regret du passé (Regret for the Past)']),
        ])
    ]
)
emotions_tree_usa_1940 = Tree(
    'Emotions (Эмоции)',
    [
        Tree('Positive (Положительные)', [
            Tree('Happiness (Радость)', ['Contentment (Удовлетворение)', 'Family Joy (Семейное счастье)', 'Personal Success (Личный успех)']),
            Tree('Love (Любовь)', ['Patriotism (Патриотизм)', 'Community Spirit (Дух сообщества)', 'Romantic Love (Романтическая любовь)']),
            Tree('Hope (Надежда)', ['Dreams of Future (Мечты о будущем)', 'Faith (Вера)'])
        ]),
        Tree('Negative (Отрицательные)', [
            Tree('Fear (Страх)', ['Worry about War (Тревога о войне)', 'Uncertainty (Неуверенность)']),
            Tree('Anger (Гнев)', ['Frustration (Раздражение)', 'Racial Tensions (Расовая напряженность)']),
            Tree('Sadness (Печаль)', ['Loss of Loved Ones (Потеря близких)', 'Financial Struggles (Финансовые трудности)'])
        ]),
        Tree('Mixed (Смешанные)', [
            Tree('Nostalgia (Ностальгия)', ['Longing for Simpler Times (Тоска по простым временам)', 'War Remembrance (Воспоминания о войне)']),
        ])
    ]
)
emotions_tree_ussr_1940 = Tree(
    'Эмоции (Emotions)',
    [
        Tree('Положительные (Positive)', [
            Tree('Радость (Joy)', ['Успех (Success)', 'Дружба (Friendship)', 'Удовольствие (Satisfaction)']),
            Tree('Любовь (Love)', ['Преданность (Devotion)', 'Сострадание (Compassion)', 'Трудовая гордость (Labor Pride)']),
            Tree('Удивление (Surprise)', ['Изумление (Amazement)', 'Восхищение (Admiration)'])
        ]),
        Tree('Отрицательные (Negative)', [
            Tree('Гнев (Anger)', ['Негодование (Indignation)', 'Ярость (Rage)', 'Зависть (Envy)']),
            Tree('Печаль (Sadness)', ['Тоска (Longing)', 'Одиночество (Loneliness)', 'Скорбь (Sorrow)']),
            Tree('Страх (Fear)', ['Тревога (Anxiety)', 'Беспокойство (Worry)', 'Опасение (Apprehension)'])
        ]),
        Tree('Смешанные (Mixed)', [
            Tree('Ностальгия (Nostalgia)', ['Тоска по прошлому (Yearning for the Past)', 'Грусть о войне (Sadness about the War)']),
        ])
    ]
)
emotions_tree_izard = Tree(
    'Эмоции (Emotions)',
    [
        Tree('Базовые эмоции (Basic Emotions)', [
            Tree('Интерес (Interest)', ['Любопытство (Curiosity)', 'Мотивация (Motivation)']),
            Tree('Радость (Joy)', ['Удовольствие (Pleasure)', 'Восторг (Delight)']),
            Tree('Удивление (Surprise)', ['Шок (Shock)', 'Изумление (Amazement)']),
            Tree('Грусть (Sadness)', ['Скорбь (Grief)', 'Тоска (Longing)']),
            Tree('Гнев (Anger)', ['Раздражение (Irritation)', 'Ярость (Rage)']),
            Tree('Отвращение (Disgust)', ['Презрение (Contempt)', 'Омерзение (Loathing)']),
            Tree('Презрение (Contempt)', ['Неприязнь (Disdain)', 'Пренебрежение (Disregard)']),
            Tree('Страх (Fear)', ['Тревога (Anxiety)', 'Паника (Panic)']),
            Tree('Стыд (Shame)', ['Вина (Guilt)', 'Унижение (Humiliation)']),
            Tree('Вина (Guilt)', ['Раскаяние (Remorse)', 'Сожаление (Regret)'])
        ]),
        Tree('Комплексные эмоции (Complex Emotions)', [
            Tree('Сочетание базовых эмоций (Combination of Basic Emotions)', [
                'Ностальгия (Nostalgia)',
                'Ревность (Jealousy)',
                'Гордость (Pride)',
                'Смущение (Embarrassment)'
            ])
        ])
    ]
)
# Draw the tree with NLTK's graphical interface
emotions_tree_izard.draw()
# emotions_tree_ussr.draw()
# emotions_tree_ussr_1940.draw()
# emotions_tree_usa_1940.draw()
# emotions_tree_france_1940.draw()
