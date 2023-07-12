class FeatureEngineer:
    def value_modifier(self, data, metric):
        if metric == 'container_memory_max_usage_bytes':
            data['values'] = data['values'] / 1048576
        elif metric == 'container_cpu_usage_seconds_total':
            data['time_diff'] = data['date_col'].diff().dt.total_seconds()
            data['usage_diff'] = data['values'].diff()
            data['utilization'] = (data['usage_diff'].diff()/data['time_diff']) * 100
        data.fillna(0, inplace=True)
        return data
            
            
    