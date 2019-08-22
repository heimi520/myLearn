
INSERT INTO MEORIENTB2B_BI.A_PRODUCER_MATCH_DEMO
SELECT 
  b.WEBSITE_ID,
	b.PURCHASER_ID,
	b.SUPPLIER_ID,
	b.MATCH_SCORE,
	b.PRODUCT_TAG_ID  AS TAG_CODE,
  TAG.T_NAME AS TAG_NAME ,
	LOCALTIMESTAMP  AS CREATE_TIME
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
                'match_already' AS match_source 
        FROM
                MEORIENTB2B_BI.A_PRODUCER_REALTIME PR
                INNER JOIN MEORIENTB2B_BI.RECOM_BUYER_FOR_SUPPLIER BS ON PR.PURCHASER_ID = BS.buyer_id 
                AND PR.WEBSITE_ID = BS.SUPPLIER_website_id 
        WHERE
                PR.ACTION_TIME >= TO_DATE( '2019-06-04 01:00:00', 'yyyy-MM-dd hh24:mi:ss' ) 
                AND PR.ACTION_TIME < TO_DATE( '2019-06-05 02:00:00', 'yyyy-MM-dd hh24:mi:ss' ) 
							  AND BS.PRODUCT_TAG_NAME IS NOT NULL
								
                UNION ALL
 
        SELECT
                PR.WEBSITE_ID ,
                PR.PURCHASER_ID,
                TSPU.SUPPLIER_ID, 
                TSPU.TAG_SCORE * 0.6 AS match_score,
								TSPU.TAG_CODE  AS  PRODUCT_TAG_ID,
                'tag_mach' AS match_source 
        FROM
                MEORIENTB2B_BI.A_PRODUCER_REALTIME PR
                INNER JOIN MEORIENTB2B_BI.RECOM_TAG_SPU_TEMP TSPU ON PR.WEBSITE_ID = TSPU.WEBSITE_ID 
                AND PR.TAG_CODE = TSPU.TAG_CODE  
        WHERE
                PR.ACTION_TIME >= TO_DATE( '2019-06-04 01:00:00', 'yyyy-MM-dd hh24:mi:ss' ) 
                AND PR.ACTION_TIME < TO_DATE( '2019-06-05 02:00:00', 'yyyy-MM-dd hh24:mi:ss' ) 

        ) a 
				)b   LEFT JOIN   MEORIENTB2B_BI.PRODUCT_TAG_DEFINE  TAG  ON b.PRODUCT_TAG_ID =TAG.TAG_CODE WHERE idx=1 
				
				
				
				
				