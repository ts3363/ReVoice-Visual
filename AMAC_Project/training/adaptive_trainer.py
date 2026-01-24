class AdaptiveTrainer:

    def __init__(self):
        self.level = 1   # 1=syllable, 2=word, 3=sentence, 4=free speech

    def get_task(self):
        if self.level == 1:
            return "Say: ba, pa, ma"
        elif self.level == 2:
            return "Say: blue bin"
        elif self.level == 3:
            return "Say: bin blue at b eight now"
        else:
            return "Free speech for 10 seconds"

    def update(self, clarity_score):
        if clarity_score > 80 and self.level < 4:
            self.level += 1
        elif clarity_score < 50 and self.level > 1:
            self.level -= 1
