from remedian import remedian



class Feature():
    def __init__(self, threshold=0.15, alpha=0.2, hard_factor=0.15, decay=0.001, max_feature_updates=0):
        self.median = remedian()
        self.min = None
        self.max = None
        self.hard_min = None
        self.hard_max = None
        self.threshold = threshold
        self.alpha = alpha
        self.hard_factor = hard_factor
        self.decay = decay
        self.last = 0
        self.current_median = 0
        self.update_count = 0
        self.max_feature_updates = max_feature_updates
        self.first_seen = -1
        self.updating = True

    def update(self, x, now=0):
        if self.max_feature_updates > 0:
            if self.first_seen == -1:
                self.first_seen = now;
        new = self.update_state(x, now=now)
        filtered = self.last * self.alpha + new * (1 - self.alpha)
        self.last = filtered
        return filtered

    def update_state(self, x, now=0):
        updating = self.updating and (self.max_feature_updates == 0 or now - self.first_seen < self.max_feature_updates)
        if updating:
            self.median + x
            self.current_median = self.median.median()
        else:
            self.updating = False
        median = self.current_median

        if self.min is None:
            if x < median and (median - x) / median > self.threshold:
                if updating:
                    self.min = x
                    self.hard_min = self.min + self.hard_factor * (median - self.min)
                return -1
            return 0
        else:
            if x < self.min:
                if updating:
                    self.min = x
                    self.hard_min = self.min + self.hard_factor * (median - self.min)
                return -1
        if self.max is None:
            if x > median and (x - median) / median > self.threshold:
                if updating:
                    self.max = x
                    self.hard_max = self.max - self.hard_factor * (self.max - median)
                return 1
            return 0
        else:
            if x > self.max:
                if updating:
                    self.max = x
                    self.hard_max = self.max - self.hard_factor * (self.max - median)
                return 1

        if updating:
            if self.min < self.hard_min:
                self.min = self.hard_min * self.decay + self.min * (1 - self.decay)
            if self.max > self.hard_max:
                self.max = self.hard_max * self.decay + self.max * (1 - self.decay)

        if x < median:
            return - (1 - (x - self.min) / (median - self.min))
        elif x > median:
            return (x - median) / (self.max - median)

        return 0
