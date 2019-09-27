

INSERT INTO MEORIENTB2B_BI.ADDED_RFQ_MATCH_DATA

SELECT
      b.rfq_id ,
      b.rfq_website_id,
      b.website_id as SUPPLIER_WEBSITE_ID,
      b.supplier_id,
      b.match_score,
      b.product_tag_id AS TAG_CODE,
      TAG.T_NAME AS TAG_NAME ,
      b.action_time ,       
      b.op_time  ,
      LOCALTIMESTAMP  AS CREATE_TIME,
	 'RFQ' AS SOURCE

FROM (

SELECT
       a.* ,row_number() over(partition by  a.rfq_website_id,a.rfq_id,a.supplier_id order by  a.match_score   desc)  as   idx
FROM
(SELECT
                PR.RFQ_WEBSITE_ID ,
                PR.RFQ_ID,
                TAGF.WEBSITE_ID ,
                TAGF.SUPPLIER_ID, 
                 100*0.6 AS match_score,
				         TAGF.PRODUCT_TAG_ID ,
                'tag_mach' AS match_source,
                PR.ACTION_TIME,
                PR.OP_TIME
        FROM
                MEORIENTB2B_BI.A_RFQ_REALTIME PR
                INNER JOIN MEORIENTB2B_BI.RECOM_SUPPLIER_PROU_TAG_FINAL  TAGF ON PR.RFQ_WEBSITE_ID = TAGF.WEBSITE_ID 
                AND PR.TAG_CODE = TAGF.PRODUCT_TAG_ID
        WHERE
                PR.OP_TIME>1548155147000 
                AND PR.OP_TIME<=1566745547000 

				    UNION ALL
 
         SELECT
                PR.RFQ_WEBSITE_ID ,
                PR.RFQ_ID,
                TAGF.WEBSITE_ID ,
                TAGF.SUPPLIER_ID, 
                 100*0.3 AS match_score,
				         TAGF.PRODUCT_TAG_ID ,
                'tag_mach' AS match_source,
                PR.ACTION_TIME,
                PR.OP_TIME
        FROM
                MEORIENTB2B_BI.A_RFQ_REALTIME PR
                INNER JOIN MEORIENTB2B_BI.RECOM_SUPPLIER_PROU_TAG_FINAL  TAGF ON PR.RFQ_WEBSITE_ID <> TAGF.WEBSITE_ID 
                AND PR.TAG_CODE = TAGF.PRODUCT_TAG_ID
        WHERE
                PR.OP_TIME>1548155147000 
                AND PR.OP_TIME<=1566745547000 
						)a
                        
                        )b   LEFT JOIN   MEORIENTB2B_BI.PRODUCT_TAG_DEFINE  TAG  ON b.PRODUCT_TAG_ID =TAG.TAG_CODE WHERE idx=1 
				