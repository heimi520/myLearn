
INSERT INTO MEORIENTB2B_BI.A_PRODUCER_MATCH_DEMO
SELECT DISTINCT
    b.WEBSITE_ID,
	b.PURCHASER_ID,
	b.SUPPLIER_ID,
	b.MATCH_SCORE,
	b.PRODUCT_TAG_ID  AS TAG_CODE,
    TAG.T_NAME AS TAG_NAME ,
    b.ACTION_TIME,
    b.OP_TIME,
	LOCALTIMESTAMP  AS CREATE_TIME,
	'PRODUCER' AS SOURCE
FROM (
SELECT
       a.* ,row_number() over(partition by  a.website_id,a.purchaser_id,a.supplier_id order by  a.match_score   desc)  as   idx
FROM
        (
        SELECT
                PR.WEBSITE_ID,
                PR.PURCHASER_ID,
                BS.SUPPLIER_ID,
                BS.MATCH_SCORE,
				BS.PRODUCT_TAG_ID  AS PRODUCT_TAG_ID,
                'match_already' AS match_source ,
                PR.ACTION_TIME,
                PR.OP_TIME
        FROM
                MEORIENTB2B_BI.A_PRODUCER_REALTIME PR
                INNER JOIN MEORIENTB2B_BI.RECOM_BUYER_FOR_SUPPLIER BS ON PR.PURCHASER_ID = BS.buyer_id 
                AND PR.WEBSITE_ID = BS.SUPPLIER_website_id 
        WHERE
                PR.ACTION_TIME > TO_DATE( '2019-06-04 01:00:00', 'yyyy-MM-dd hh24:mi:ss' ) 
                AND PR.ACTION_TIME <= TO_DATE( '2019-06-05 02:00:00', 'yyyy-MM-dd hh24:mi:ss' )
        
                AND BS.PRODUCT_TAG_NAME IS NOT NULL
								
                UNION ALL
 
        SELECT
                PR.WEBSITE_ID ,
                PR.PURCHASER_ID,
                TAGF.SUPPLIER_ID, 
                 100*0.6 AS match_score,
				         TAGF.PRODUCT_TAG_ID ,
                'tag_match' AS match_source,
                PR.ACTION_TIME,
                PR.OP_TIME
        FROM
                MEORIENTB2B_BI.A_PRODUCER_REALTIME PR
                INNER JOIN MEORIENTB2B_BI.RECOM_SUPPLIER_PROU_TAG_FINAL  TAGF ON PR.WEBSITE_ID = TAGF.WEBSITE_ID 
                AND PR.TAG_CODE = TAGF.PRODUCT_TAG_ID
        WHERE
                PR.ACTION_TIME > TO_DATE( '2019-06-04 01:00:00', 'yyyy-MM-dd hh24:mi:ss' ) 
                AND PR.ACTION_TIME <= TO_DATE( '2019-06-05 02:00:00', 'yyyy-MM-dd hh24:mi:ss' ) 
        
								
								
				    UNION ALL
 
        SELECT
                PR.WEBSITE_ID ,
                PR.PURCHASER_ID,
                T2F.SUPPLIER_ID, 
                 50*0.6 AS match_score,
				         T2F.PRODUCT_TAG_ID  ,
                'tag_match' AS match_source,
                PR.ACTION_TIME,
                PR.OP_TIME
        FROM
                MEORIENTB2B_BI.A_PRODUCER_REALTIME PR
                INNER JOIN MEORIENTB2B_BI.RECOM_SUPPLIER_T2_TAG_FINAL T2F ON PR.WEBSITE_ID = T2F.WEBSITE_ID 
                AND PR.TAG_CODE = T2F.PRODUCT_TAG_ID
        WHERE
                PR.ACTION_TIME > TO_DATE( '2019-06-04 01:00:00', 'yyyy-MM-dd hh24:mi:ss' ) 
                AND PR.ACTION_TIME <= TO_DATE( '2019-06-05 02:00:00', 'yyyy-MM-dd hh24:mi:ss' ) 
            
												
							
        ) a 
				)b   LEFT JOIN   MEORIENTB2B_BI.PRODUCT_TAG_DEFINE  TAG  ON b.PRODUCT_TAG_ID =TAG.TAG_CODE WHERE idx=1 
				
				