import pandas as pd
from supabase import create_client

class SupabaseClient():
    '''
    wrapper for supabase client that downloads the table
    '''
    def __init__(self, url, key, table):
        self.client = create_client(
            url, key
        ).table(table)
        self.data = None
    
    def get_table_count(self):
        '''
        Gets the number of rows in the table, which is necessary for
        pagination
        '''
        resp = self.client.select(
            '*', count="exact"
        ).execute()
        ## return ##
        return resp.count
    
    def offset_req(self, start, end):
        '''
        Makes and offset request of the table with a start and finish
        for pagination
        Ranges are 0 indexed and inclusive on start, but not end
        '''
        resp = self.client.select('*').range(start, end).execute()
        ## return ##
        return resp.data
    
    def get_data(self):
        '''
        Returns the table using pagination
        '''
        ## container of processed rows ##
        dfs = []
        ## counter ##
        processed_rows = 0
        ## get row count ##
        rows = self.get_table_count()
        ## paginate ##
        while processed_rows < rows:
            ## set range start ##
            index_start = processed_rows ## note, b/c it is zero indexed, its not +1
            ## get end range ##
            index_end = (
                index_start + 1000 
                if index_start + 1000 < rows
                else rows
            )
            ## get data ##
            data = self.offset_req(index_start, index_end)
            ## append to frames ##
            dfs.append(pd.DataFrame(data))
            ## update processed rows ##
            processed_rows += len(data)
        ## combine ##
        df = pd.concat(dfs)
        ## store
        self.data = df.sort_values(
            by=['game_id'],
            ascending=[True]
        ).reset_index(drop=True)
        ## return ##
        return self.data
