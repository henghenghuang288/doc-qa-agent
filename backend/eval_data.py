"""
评估数据集(Golden Set)

按照FDE岗位标准做法构建:一份标准测试文档 + 一组标注好预期结果的测试题。
题目分两类:
  - answerable: 文档里确实有答案,检查回答是否命中关键信息、grounded 是否为真
  - unanswerable: 文档里没有答案(含"诱导编造"陷阱题,故意问得像有答案一样),
    检查系统是否正确拒答而不是编造

这份文档故意设计成中等复杂度的企业服务类内容,贴近"询盘自动回复/产品咨询客服"场景,
覆盖价格、政策、技术规格、合作流程等真实客服高频问题类型。
"""

GOLDEN_DOCUMENT = """公司简介
云策科技成立于2022年,专注于为中小外贸企业提供独立站建站与询盘管理SaaS服务。
总部位于深圳,团队规模约45人,目前服务约800家活跃客户。

产品矩阵
我们提供三条产品线:独立站建站工具"云策建站"、询盘自动分配系统"云策CRM"、
多语言客服机器人"云策客服"。三款产品可单独购买,也可组合订阅享受套餐折扣。

价格与套餐
基础版:每月299元,包含1个站点、500条询盘额度、基础客服机器人。
专业版:每月899元,包含3个站点、5000条询盘额度、多语言客服机器人、专属客户经理。
企业版:价格面议,支持私有化部署、API对接、定制开发,起步规模通常50个站点以上。
所有套餐均提供15天免费试用,试用期内可随时取消,不收取任何费用。

询盘自动分配规则
系统根据询盘语言、产品类别、客户历史成交记录三个维度自动分配给对应销售。
分配延迟不超过30秒。如果30分钟内销售未响应,系统会自动升级提醒给销售主管。

技术支持与SLA
专业版及以上客户提供工作日9:00-21:00在线支持,企业版客户额外提供7x24小时电话支持。
系统可用性承诺为99.5%,如未达标当月费用按比例返还。

数据安全
所有客户数据存储在国内服务器,符合《数据安全法》要求。企业版客户可选择私有化部署,
数据完全不出客户自己的服务器。我们通过了等保三级认证。

退款政策
按月订阅可随时取消,取消后当月已使用部分不退款,下月不再扣费。
按年订阅的客户,如在使用不满3个月内申请退款,按剩余月份的80%退还。

合作伙伴计划
我们与12家外贸服务机构建立了渠道合作,合作伙伴可获得首年订阅15%的返佣。
申请合作需要满足年服务客户数不低于50家的门槛。

近期更新
2026年5月,云策客服上线了基于大模型的智能问答能力,可以根据客户上传的产品手册自动学习产品知识。
2026年3月,云策CRM新增了WhatsApp渠道询盘接入。"""


GOLDEN_QUESTIONS = [
    # ---- 可回答:基础事实题 ----
    {"id": "q1", "question": "专业版多少钱一个月", "answerable": True, "expect_keywords": ["899"]},
    {"id": "q2", "question": "基础版包含多少条询盘额度", "answerable": True, "expect_keywords": ["500"]},
    {"id": "q3", "question": "有没有免费试用", "answerable": True, "expect_keywords": ["15天", "试用"]},
    {"id": "q4", "question": "询盘多久能分配给销售", "answerable": True, "expect_keywords": ["30秒"]},
    {"id": "q5", "question": "系统可用性承诺是多少", "answerable": True, "expect_keywords": ["99.5"]},
    {"id": "q6", "question": "数据存储在哪里安全吗", "answerable": True, "expect_keywords": ["国内服务器", "数据安全法"]},
    {"id": "q7", "question": "公司是哪一年成立的", "answerable": True, "expect_keywords": ["2022"]},
    {"id": "q8", "question": "按年订阅不满3个月退款怎么算", "answerable": True, "expect_keywords": ["80%"]},
    {"id": "q9", "question": "合作伙伴返佣比例是多少", "answerable": True, "expect_keywords": ["15%"]},
    {"id": "q10", "question": "企业版有没有电话支持", "answerable": True, "expect_keywords": ["7x24", "电话"]},
    {"id": "q11", "question": "公司团队规模多大", "answerable": True, "expect_keywords": ["45"]},
    {"id": "q12", "question": "云策客服最近有什么新功能", "answerable": True, "expect_keywords": ["大模型", "智能问答"]},
    {"id": "q13", "question": "申请成为合作伙伴有什么门槛", "answerable": True, "expect_keywords": ["50家"]},
    {"id": "q14", "question": "如果30分钟销售没回复会怎样", "answerable": True, "expect_keywords": ["升级", "主管"]},
    {"id": "q15", "question": "有没有通过安全认证", "answerable": True, "expect_keywords": ["等保三级"]},

    # ---- 不可回答:文档里确实没有 ----
    {"id": "q16", "question": "你们CEO是谁", "answerable": False},
    {"id": "q17", "question": "支持哪些支付方式", "answerable": False},
    {"id": "q18", "question": "公司去年营收多少", "answerable": False},
    {"id": "q19", "question": "和Shopify相比你们优势在哪", "answerable": False},
    {"id": "q20", "question": "有没有Android手机APP", "answerable": False},

    # ---- 诱导编造陷阱题:问法像有答案,实际文档没写,专门测试是否会瞎编 ----
    {"id": "q21", "question": "基础版的客服机器人支持几种语言", "answerable": False,
     "note": "文档只说专业版是'多语言客服机器人',没说基础版支持几种语言,容易被诱导编出一个数字"},
    {"id": "q22", "question": "企业版具体多少钱", "answerable": False,
     "note": "文档明确写'价格面议',容易被诱导编一个具体数字"},
    {"id": "q23", "question": "退款一般几天到账", "answerable": False,
     "note": "文档只说怎么算退款比例,没说到账时效,容易被诱导编一个天数"},
    {"id": "q24", "question": "WhatsApp渠道是什么时候上线的,具体哪一天", "answerable": False,
     "note": "文档只写了'2026年3月',没写具体哪一天,容易被诱导编出精确日期"},
]
