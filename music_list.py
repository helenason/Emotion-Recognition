class Music() :
    @staticmethod
    def print_music(first_res, second_res) :
        if first_res == 0 : # anger
            music = [
                {'Name' : 'abcdefu',
                'Artist' : 'Gayle'},
                {'Name' : 'WTF',
                'Artist' : 'Hugel'},
                {'Name' : '16 shots',
                'Artist' : 'Stefflon Don'},
                {'Name' : 'New thing',
                'Artist' : 'ZICO'},
                {'Name' : "Ain't my fault",
                'Artist' : 'Zara Larsson'}
            ]
        elif first_res == 1 : # disgust
            music = [
                {'Name' : 'She',
                'Artist' : 'Winona Oak'},
                {'Name' : 'Psycho',
                'Artist' : 'Mia Rodriguez'},
                {'Name' : "I feel like I'm drowning",
                'Artist' : 'Two Feet'},
                {'Name' : 'Hynotic',
                'Artist' : 'Vanic X Zella Day'},
                {'Name' : 'Savage',
                'Artist' : 'Bahari'}
            ]
        elif first_res == 2 : # fear
            music = [
                {'Name' : 'King',
                'Artist' : 'Zayde Wolf'},
                {'Name' : 'Shakedown',
                'Artist' : 'Score'},
                {'Name' : 'Salt',
                'Artist' : 'Ava Max'},
                {'Name' : "I'm back",
                'Artist' : 'Royal Deluxe'},
                {'Name' : 'only one',
                'Artist' : 'The score'}
            ]
        elif first_res == 3 : # happy
            music = [
                {'Name' : 'By My Side',
                'Artist' : 'J.BASS'},
                {'Name' : 'inside',
                'Artist' : 'JUNNY'},
                {'Name' : '3 2 1',
                'Artist' : 'JAEHA'},
                {'Name' : 'favorite things',
                'Artist' : 'HAAN'},
                {'Name' : 'Cupid',
                'Artist' : 'fifty fifty'}
            ]
        elif first_res == 4 : # sad
            music = [
                {'Name' : 'Love Poem',
                'Artist' : 'IU'},
                {'Name' : 'Gravity',
                'Artist' : 'TAEYEON'},
                {'Name' : '12:45',
                'Artist' : 'Etham'},
                {'Name' : 'Dangerously',
                'Artist' : 'Charlie Puth'},
                {'Name' : 'Falling',
                'Artist' : 'Harry Styles'}
            ]
        elif first_res == 5 : # surprise
            music = [
                {'Name' : 'Letters',
                'Artist' : 'Maximillan'},
                {'Name' : 'Tell a son',
                'Artist' : 'Peder Elias'},
                {'Name' : 'Never not',
                'Artist' : 'Lauv'},
                {'Name' : 'Paradise',
                'Artist' : 'Pink Sweat$'},
                {'Name' : 'anything 4 u',
                'Artist' : 'LANY'}
            ]
        else : # neutral
            music = Music.print_music(second_res, -1) # 두번째 결과로
        
        return music