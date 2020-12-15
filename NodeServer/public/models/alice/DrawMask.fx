// パラメータ宣言

// 座法変換行列
float4x4 WorldViewProjMatrix      : WORLDVIEWPROJECTION;

///////////////////////////////////////////////////////////////////////////////////////////////
// オブジェクト描画

struct VS_OUTPUT2
{
    float4 Pos        : POSITION;    // 射影変換座標
    float  Col 		  : TEXCOORD0;    // マスク値
};

// 頂点シェーダ
VS_OUTPUT2 Mask_VS(float4 Pos : POSITION, float3 Normal : NORMAL, float2 Tex : TEXCOORD0)
{
    VS_OUTPUT2 Out = (VS_OUTPUT2)0;
    
    // カメラ視点のワールドビュー射影変換
    Out.Pos = mul( Pos, WorldViewProjMatrix );
    return Out;
}

// ピクセルシェーダ
float4 Mask_PS( VS_OUTPUT2 IN ) : COLOR0
{
    return float4(0,0,0,0);
}

// オブジェクト描画用テクニック
technique MaskTec < string MMDPass = "object"; > {
    pass DrawObject {
        AlphaBlendEnable = false;
        AlphaTestEnable = false;
        VertexShader = compile vs_2_0 Mask_VS();
        PixelShader  = compile ps_2_0 Mask_PS();
    }
}

// オブジェクト描画用テクニック
technique MaskTecBS  < string MMDPass = "object_ss"; > {
    pass DrawObject {
        AlphaBlendEnable = false;
        AlphaTestEnable = false;
        VertexShader = compile vs_2_0 Mask_VS();
        PixelShader  = compile ps_2_0 Mask_PS();
    }
}
technique EdgeTec < string MMDPass = "edge"; > { }
technique ShadowTech < string MMDPass = "shadow";  > { }
technique ZplotTec < string MMDPass = "zplot"; > { }

///////////////////////////////////////////////////////////////////////////////////////////////
