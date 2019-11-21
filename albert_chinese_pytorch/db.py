import sqlite3
import os
import Terry_toolkit as tkit

class NodesDb:
    def __init__(self):
     
        self.DB ='dataset/node.db'  
        """
        连接数据库
        """


        if os.path.exists( self.DB ):
            print("数据库已经存在")
            self.conn = sqlite3.connect(self.DB)
            self.connect = self.conn.cursor()
        else:
            try:
                self.create_table()
            except:
                pass
        # self.connect =
    def close(self):
        """
        关闭数据库
        """
        self.conn.close()

    def create_table(self):
        # conn = sqlite3.connect(self.DB)
        # c = conn.cursor()
        # Create table
        #创建链接表
        self.conn = sqlite3.connect(self.DB)
        self.connect = self.conn.cursor()
        self.connect.execute('''CREATE TABLE `keywords` ( `id` INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,`keyword` TEXT )''')
        self.connect.execute('''CREATE TABLE `nodes` ( `id` INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE, `title` TEXT, `content` TEXT, `author` INTEGER, `url` INTEGER )''')
        self.conn.commit()
        pass

    def add_keywords(self,keywords):
        # self.connect
        sql="INSERT INTO keywords VALUES (?,?)"
        # ls=[]
        for item in keywords:
            if self.check_keyword(item):
                pass
            else:
                self.connect.executemany(sql,[(None,item)])
                self.conn.commit()      
    def add_nodes(self,nodes):
        """
        添加数据
        """
        # self.connect
        sql="INSERT INTO nodes VALUES (?,?,?,?,?)"
        # ls=[]
        texts=[]
        for i,nitem in enumerate(nodes):
            # self.connect.executemany(sql,[(None,item,rank,keyword)])
            # self.conn.commit() 
            title,content,author,url=nitem 
            if self.check_node(url):
                print("跳过: ",title,url)
                pass
            else:
                print("添加: ",title,url)
                texts.append((None, title,content,author,url))
                if i%10000==0 :
                    self.connect.executemany(sql,texts)
                    self.conn.commit() 
                    texts=[] 
        self.connect.executemany(sql,texts)
        self.conn.commit() 
    def check_keyword(self,keyword):

        sql= "select * from keywords where keywords.keyword='"+keyword+"'"
        # print(sql)
        # conn.commit()
        self.connect.execute(sql)
        one=  self.connect.fetchone()
        # print(one)
        # conn.close()
        return one
    def check_node(self,url):
        """
        检查是否存在
        """
        sql= "select * from nodes where nodes.url='"+url+"'"
        # print(sql)
        self.connect.execute(sql)
        one=  self.connect.fetchone()
        return one

    def get_keywords(self,limit=1000):
        """
        随机获取一千个关键词
        """
        sql="SELECT * FROM keywords ORDER BY RANDOM() limit "+str(limit)
        self.connect.execute(sql)
        return self.connect.fetchall()

    def get_nodes(self,limit=1000):
        """
        随机获取一千个关键词
        """
        sql="SELECT * FROM nodes where nodes.label notNULL  limit "+str(limit)
        self.connect.execute(sql)
        return self.connect.fetchall()
    def get_unlabel_nodes(self,limit=2):
        """
        获取未标记的数据
        """
        sql="SELECT * FROM nodes where nodes.label isnull  ORDER BY RANDOM()  limit "+str(limit)
        self.connect.execute(sql)
        return self.connect.fetchall()
    def set_unlabel_nodes(self,id,label):
        """
        修改标记
        UPDATE COMPANY SET ADDRESS = 'Texas', SALARY = 20000.00;
        """
        sql="UPDATE nodes set label="+str(label)+" where nodes.id="+str(id)
        print(sql)
        try:
            self.connect.execute(sql)
            self.conn.commit() 
            return True
        except:
            return False
        # return self.connect.fetchall()
def bulid():
    db=NodesDb()
    # tjson=tkit.Json(file_path="train.json")
    data=[]
    for  id,title,content,author,url,label in db.get_nodes(limit=1000000000000):
        # print(node[5])
        # print(node[5])
        # print(label)
        data.append({'label':label, 'sentence':title+"\n"+content})



    print("train",len(data[0:int(len(data)*0.8)]))
    print("dev",len(data[int(len(data)*0.8):int(len(data)*0.9)]))
    print("test",len(data[int(len(data)*0.9):]))
    # print("8",len(data[::int(100.0/10)]))
    tjson=tkit.Json(file_path="dataset/train.json")
    tjson.save(data[0:int(len(data)*0.8)])

    tjson1=tkit.Json(file_path="dataset/dev.json")
    tjson1.save(data[int(len(data)*0.8):int(len(data)*0.9)])

    tjson2=tkit.Json(file_path="dataset/test.json")
    tjson2.save(data[int(len(data)*0.9):])
# db=NodesDb()
# print(db.get_unlabel_nodes())
if __name__ == "__main__":
    # text="养龟之前要想好的问题\n想养龟之前先问自己能否承受寂寞。恐怕没有人会想到十年后，二十年，甚至五十年后的今天这条龟降临何方。龟龟虽小，养起来真是一个足够大的包袱，没有充分的思想准备。理解龟龟的生长其实养龟最辛苦的时期是小龟饲养后的第二年，第一年一般兴趣高涨，对于龟龟的照顾无微不至，看着龟龟生长迅速，看着龟龟天天长大，心里的满足溢于言表，但养了几年后，龟龟的生长开始变缓，体长每月1公分甚至不到的变化，经常使龟友开始耐不住寂寞，开始埋怨生长的缓慢，其时兴趣与耐心正在一点点失去。但在这个时期，自己的爱龟刚刚成年，正在向成熟进发，爱龟的一天天的熟悉自己，幼稚的身躯慢慢挂上凝重的颜色，如此的过程不正是养龟人期待的过程，不正是成功的体验？不要被别人的意见所左右很多朋友将自己的龟贴出来供大家赏评，赞美的话是最喜欢听的（无论谁，除非是受虐狂），但龟友很多意见有好自然也有不好，要有心里承受能力，否则，最好不贴，别人一说这是烂甲，这品相不好，这是隆背，马上对爱龟的热爱从顶峰掉到谷底，回家怎么看自己的龟龟怎么不顺眼，完了，不久的将来，这条龟一定不存在于这个窝。养龟不是为别人养的，自己的龟只属于自己，让别人说去吧，能把一条巴西养十年以上值得佩服。喜新厌旧要不得交流的时间多了，自己的鉴赏力提高，也见过很多好龟，于是缅陆龟养一段时间换印星，印星换小苏，小苏换靴脚，又养缅星，还有更好的辐射，喜新厌旧使自己变成了一个快乐的饲养员，不停的饲养着，没有一刻的宁静，也慢慢消耗了自己的耐心，因为没有人喜欢终日耕种没有收获，养龟的乐趣是在漫长龟龟成熟的过程中体验。图片来源于网络"
    bulid()