class Request:
    def __init__(self):
        self.success = 0
        self.fail = 0

    def increment_success(self):
        self.success = self.success + 1

    def increment_fail(self):
        self.fail = self.fail + 1

    # def __repr__(self):
    #     return 'success=' + str(self.success) + ', failure=' + str(self.fail) + ', total=' + str(self.success + self.fail)
    #
    def __repr__(self):
        # return str(self.success) + ' / ' + str(self.fail) + ' / ' + str(self.success + self.fail)
        return str(self.success) + ' / ' + str(self.success + self.fail) + " = " + str(round(self.success / (self.success + self.fail) * 100, 2)) + "%"


